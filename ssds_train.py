from __future__ import print_function
import numpy as np
import os
import sys
import cv2
import random
import pickle

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torch.nn.init as init

from tensorboardX import SummaryWriter

from layers import *
from layers.modules.multibox_loss import MultiBoxLoss2
from data import BaseTransform
from utils.timer import Timer
from data.data_augment import preproc
from data.dataset_factory import load_data
from utils.config import cfg
from layers.functions import Detect, PriorBox
from layers.functions.detection import Detect2
from models import *
#from utils.eval_utils import *
#from utils.visualize_utils import *

class Solver(object):
    """
    A wrapper class for the training process
    """
    def __init__(self):
        self.cfg = cfg

        # Load data
        print('===> Loading data')
        self.train_loader = load_data(cfg.dataset, 'train') if 'train' in cfg.phase else None
        self.eval_loader = load_data(cfg.dataset, 'eval') if 'eval' in cfg.phase else None
        self.test_loader = load_data(cfg.dataset, 'test') if 'test' in cfg.phase else None
        # self.visualize_loader = load_data(cfg.DATASET, 'visualize') if 'visualize' in cfg.PHASE else None

        # Build model
        print('===> Building model')
        self.base_trans = BaseTransform(cfg.image_size[0], cfg.network.rgb_means, cfg.network.rgb_std, (2, 0, 1))
        self.priors = PriorBox(cfg.anchor)
        self.model = eval(cfg.model+'.build_net')(cfg.image_size[0], cfg.dataset.num_classes)
        with torch.no_grad():
            self.priors = self.priors.forward()
        self.detector = Detect2(cfg.post_process)
        # Utilize GPUs for computation
        self.use_gpu = torch.cuda.is_available()
        if cfg.train.train_scope == '':
            trainable_param = self.model.parameters()
        else:
            trainable_param = self.trainable_param(cfg.train.train_scope)
        self.output_dir = os.path.join(cfg.output_dir, cfg.name, cfg.date)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.log_dir = os.path.join(self.output_dir, 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.checkpoint = cfg.train.checkpoint

        previous = self.find_previous()
        previous = False
        if previous:
            self.start_epoch = previous[0][-1]
            self.resume_checkpoint(previous[1][-1])
        else:
            self.start_epoch = self.initialize()
        if self.use_gpu:
            print('Utilize GPUs for computation')
            print('Number of GPU available', torch.cuda.device_count())
            self.model.cuda()
            self.priors.cuda()
            cudnn.benchmark = True
            if cfg.ngpu > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=list(range(cfg.ngpu)))
        # Print the model architecture and parameters
        #print('Model architectures:\n{}\n'.format(self.model))

        #print('Parameters and size:')
        #for name, param in self.model.named_parameters():
        #    print('{}: {}'.format(name, list(param.size())))
        # print trainable scope
        print('Trainable scope: {}'.format(cfg.train.train_scope))
        self.optimizer = self.configure_optimizer(trainable_param, cfg.train.optimizer)
        self.exp_lr_scheduler = self.configure_lr_scheduler(self.optimizer, cfg.train.lr_scheduler)
        self.max_epochs = cfg.train.lr_scheduler.max_epochs
        # metric
        if cfg.network.multi_box_loss_type == 'origin':
            self.criterion = MultiBoxLoss2(cfg.matcher, self.priors, self.use_gpu)
        else:
            print('ERROR: '+cfg.multi_box_loss_type+' is not supported')
            sys.exit()
        # Set the logger
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.checkpoint_prefix = cfg.name+'_'+cfg.dataset.dataset


    def save_checkpoints(self, epochs, iters=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if iters:
            filename = self.checkpoint_prefix + '_epoch_{:d}_iter_{:d}'.format(epochs, iters) + '.pth'
        else:
            filename = self.checkpoint_prefix + '_epoch_{:d}'.format(epochs) + '.pth'
        filename = os.path.join(self.output_dir, filename)
        torch.save(self.model.state_dict(), filename)
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'a') as f:
            f.write('epoch {epoch:d}: {filename}\n'.format(epoch=epochs, filename=filename))
        print('Wrote snapshot to: {:s}'.format(filename))

        # TODO: write relative cfg under the same page

    def resume_checkpoint(self, resume_checkpoint):
        if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
            return False
        print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        checkpoint = torch.load(resume_checkpoint)

        # print("=> Weigths in the checkpoints:")
        # print([k for k, v in list(checkpoint.items())])

        # remove the module in the parrallel model
        if 'module.' in list(checkpoint.items())[0][0]:
            pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
            checkpoint = pretrained_dict

        resume_scope = self.cfg.train.resume_scope
        # extract the weights based on the resume scope
        if resume_scope != '':
            pretrained_dict = {}
            for k, v in list(checkpoint.items()):
                for resume_key in resume_scope.split(','):
                    if resume_key in k:
                        pretrained_dict[k] = v
                        break
            checkpoint = pretrained_dict

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in self.model.state_dict()}
        # print("=> Resume weigths:")
        # print([k for k, v in list(pretrained_dict.items())])

        checkpoint = self.model.state_dict()

        unresume_dict = set(checkpoint)-set(pretrained_dict)
        if len(unresume_dict) != 0:
            print("=> UNResume weigths:")
            print(unresume_dict)

        checkpoint.update(pretrained_dict)

        return self.model.load_state_dict(checkpoint)


    def find_previous(self):
        if not os.path.exists(os.path.join(self.output_dir, 'checkpoint_list.txt')):
            return False
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'r') as f:
            lineList = f.readlines()
        epoches, resume_checkpoints = [list() for _ in range(2)]
        for line in lineList:
            epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
            checkpoint = line[line.find(':') + 2:-1]
            epoches.append(epoch)
            resume_checkpoints.append(checkpoint)
        return epoches, resume_checkpoints

    def weights_init(self, m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


    def initialize(self):
        # TODO: ADD INIT ways
        # raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")
        # for module in self.cfg.TRAIN.TRAINABLE_SCOPE.split(','):
        #     if hasattr(self.model, module):
        #         getattr(self.model, module).apply(self.weights_init)
        if self.checkpoint:
            print('Loading initial model weights from {:s}'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)
            return cfg.train.resume_epoch
        else:
            self.model.init_model(cfg.network.basenet)
            return 0


    def trainable_param(self, trainable_scope):
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_param = []
        for module in trainable_scope.split(','):
            if hasattr(self.model, module):
                # print(getattr(self.model, module))
                for param in getattr(self.model, module).parameters():
                    param.requires_grad = True
                trainable_param.extend(getattr(self.model, module).parameters())

        return trainable_param

    def train_model(self):

        # export graph for the model, onnx always not works
        # self.export_graph()

        # warm_up epoch
        for epoch in iter(range(self.start_epoch+1, self.max_epochs+1)):
            #learning rate
            sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.max_epochs))
            self.exp_lr_scheduler.step(epoch-cfg.train.lr_scheduler.warmup)
            if 'train' in cfg.phase:
                self.train_epoch(self.model, self.train_loader, self.optimizer, self.criterion, self.writer, epoch, self.use_gpu)
            if 'eval' in cfg.phase and epoch%cfg.test_frequency == 0:
                self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, epoch, self.use_gpu)
            #if 'test' in cfg.PHASE:
            #    self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir, self.use_gpu)
            #if 'visualize' in cfg.PHASE:
            #    self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)

            if epoch % cfg.train.save_frequency == 0:
                self.save_checkpoints(epoch)


    def train_epoch(self, model, data_loader, optimizer, criterion, writer, epoch, use_gpu):
        model.train()

        epoch_size = len(data_loader)
        batch_iterator = iter(data_loader)

        loc_loss = 0
        conf_loss = 0
        _t = Timer()

        for iteration in iter(range((epoch_size))):
            with torch.no_grad():
                images, targets = next(batch_iterator)
                if use_gpu:
                    images = images.cuda()
                    targets = [anno.cuda() for anno in targets]
            _t.tic()
            # forward
            out = model(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)

            # some bugs in coco train2017. maybe the annonation bug.
            if loss_l.item() == float("Inf"):
                continue

            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()

            time = _t.toc()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            # log per iter
            log = '\r==>Train: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.item(), cls_loss=loss_c.item())

            sys.stdout.write(log)
            sys.stdout.flush()

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        lr = optimizer.param_groups[0]['lr']
        log = '\r==>Train: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || lr: {lr:.6f}\n'.format(lr=lr,
                time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
        sys.stdout.write(log)
        sys.stdout.flush()

        # log for tensorboard
        writer.add_scalar('Train/loc_loss', loc_loss/epoch_size, epoch)
        writer.add_scalar('Train/conf_loss', conf_loss/epoch_size, epoch)
        writer.add_scalar('Train/lr', lr, epoch)

    def eval_epoch(self, model, data_loader, detector, output_dir, use_gpu):

        model.eval()
        dataset = data_loader.dataset
        num_images = len(testset)
        num_classes =  cfg.dataset.num_classes
        all_boxes = [[[] for _ in range(num_images)]
                    for _ in range(num_classes)]

        _t = {'im_detect': Timer(), 'misc': Timer()}
        det_file = os.path.join(self.output_dir, 'detections.pkl')

        if cfg.test.retest:
            f = open(det_file, 'rb')
            all_boxes = pickle.load(f)
            print('Evaluating detections')
            testset.evaluate_detections(all_boxes, save_folder)
            return

        for i in range(num_images):
            img = testset.pull_image(i)
            with torch.no_grad():
                x = transform(img).unsqueeze(0)
            if cuda:
                x = x.to(torch.device("cuda"))

            _t['im_detect'].tic()
            out = net(x=x, test=True)  # forward pass
            boxes, scores = detector.forward(out, self.priors)
            detect_time = _t['im_detect'].toc()
            boxes = boxes[0]
            scores = scores[0]

            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            # scale each detection back up to the image
            scale = torch.Tensor([img.shape[1], img.shape[0],
                                img.shape[1], img.shape[0]]).cpu().numpy()
            boxes *= scale

            _t['misc'].tic()

            for j in range(1, num_classes):
                inds = np.where(scores[:, j] > cfg.post_process.score_threshold)[0]
                if len(inds) == 0:
                    all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = boxes[inds]
                c_scores = scores[inds, j]
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)
                keep = nms(c_dets, cfg.post_process.nms, force_cpu=False)
                c_dets = c_dets[keep, :]
                all_boxes[j][i] = c_dets
            if cfg.post_process.max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            nms_time = _t['misc'].toc()

            if i % 20 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                    .format(i + 1, num_images, detect_time, nms_time))
                _t['im_detect'].clear()
                _t['misc'].clear()

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        if args.dataset == 'VOC':
            APs, mAP = testset.evaluate_detections(all_boxes, save_folder)
        else:
            testset.evaluate_detections(all_boxes, save_folder)


    def configure_optimizer(self, trainable_param, cfg):
        if cfg.optimizer == 'sgd':
            optimizer = optim.SGD(trainable_param, lr=cfg.lr,
                        momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(trainable_param, lr=cfg.lr,
                        momentum=cfg.momentum, alpha=cfg.alpha, eps=cfg.eps, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'adam':
            optimizer = optim.Adam(trainable_param, lr=cfg.lr,
                        betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.weight_decay)
        else:
            AssertionError('optimizer can not be recognized.')
        return optimizer


    def configure_lr_scheduler(self, optimizer, cfg):
        if cfg.lr_decay_type == 'multi-step':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.steps, gamma=cfg.gamma)
        elif cfg.lr_decay_type == 'exponential':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)
        elif cfg.lr_decay_type == 'cos':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
        else:
            AssertionError('scheduler can not be recognized.')
        return scheduler

    #TODO: export graph
    def export_graph(self):
        pass


def train_model():
    s = Solver()
    s.train_model()
    return True

