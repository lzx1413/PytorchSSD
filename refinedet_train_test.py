from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot,VOC_320, AnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, preproc,VOC_512
#from layers.modules import MultiBoxLoss
from layers.modules import RefineMultiBoxLoss
from layers.functions import Detect,PriorBox
import time
from utils.nms_wrapper import nms
from utils.timer import Timer
import pickle

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='Refine_vgg',
                    help='Refine_vgg')
parser.add_argument('-s', '--size', default='320',
                    help='320 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='/mnt/lvmhdd1/zuoxin/ssd_pytorch_models/vgg16_reducedfc.pth', help='pretrained base model')
#parser.add_argument(
#    '--basenet', default='/mnt/lvmhdd1/zuoxin/ssd_pytorch_models/mb.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--gpu_id', default=[0,1], type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--resume_net', default=False, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')

parser.add_argument('-max','--max_epoch', default=300,
                    type=int, help='max epoch for retraining')
parser.add_argument('-we','--warm_epoch', default=1,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='/mnt/lvmhdd1/zuoxin/ssd_pytorch_models/refine/',
                    help='Location to save checkpoint models')
parser.add_argument('--date',default='0327')
parser.add_argument('--save_frequency',default=10)
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--test_frequency',default=10)
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
args = parser.parse_args()

save_folder = os.path.join(args.save_folder,args.version+'_'+args.size,args.date)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
test_save_dir = os.path.join(save_folder,'ss_predict')
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)

log_file_path = save_folder + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_320, VOC_512)[args.size == '512']
else:
    train_sets = [('2014', 'train'),('2014', 'valminusminival')]
    cfg = (VOC_320, VOC_512)[args.size == '512']

img_dim = (320,512)[args.size=='512']
rgb_std = (1,1,1)
if 'vgg' in args.version:
    rgb_means = (104, 117, 123)
p = (0.6,0.2)[args.version == 'RFB_mobile']
num_classes = (21, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9
if args.visdom:
    import visdom
    viz = visdom.Visdom()

from models.RefineSSD_vgg import build_net
cfg = VOC_320
net = build_net(320, num_classes,use_refine=True)
print(net)
if not args.resume_net:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    net.base.load_state_dict(base_weights)

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    print('Initializing weights...')
# initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.trans_layers.apply(weights_init)
    net.latent_layrs.apply(weights_init)
    net.up_layers.apply(weights_init)
    net.arm_loc.apply(weights_init)
    net.arm_conf.apply(weights_init)
    net.odm_loc.apply(weights_init)
    net.odm_conf.apply(weights_init)
else:
# load resume network
    resume_net_path = os.path.join(save_folder,args.version+'_'+args.dataset + '_epoches_'+ \
                           str(args.resume_epoch) + '.pth')
    print('Loading resume network',resume_net_path)
    state_dict = torch.load(resume_net_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.gpu_id:
    net = torch.nn.DataParallel(net, device_ids=args.gpu_id)

if args.cuda:
    net.cuda()
    cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
#optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                     momentum=args.momentum, weight_decay=args.weight_decay)

arm_criterion = RefineMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False)
odm_criterion = RefineMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False,0.01)
priorbox = PriorBox(cfg)
detector = Detect(num_classes,0,cfg,object_score=0.01)
priors = Variable(priorbox.forward(), volatile=True)
#dataset
print('Loading Dataset...')
if args.dataset == 'VOC':
    testset = VOCDetection(
        VOCroot, [('2007', 'test')], None, AnnotationTransform())
    train_dataset = VOCDetection(VOCroot, train_sets, preproc(
        img_dim, rgb_means, p), AnnotationTransform())
elif args.dataset == 'COCO':
    testset = COCODetection(
        COCOroot, [('2014', 'minival')], None)
    train_dataset = COCODetection(COCOroot, train_sets, preproc(
        img_dim, rgb_means, p))
else:
    print('Only VOC and COCO are supported now!')
    exit()


def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    if args.resume_net:
        epoch = 0 + args.resume_epoch
    epoch_size = len(train_dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC,stepvalues_COCO)[args.dataset=='COCO']
    print('Training',args.version, 'on', train_dataset.name)
    step_index = 0

    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    log_file = open(log_file_path,'w')
    batch_iterator = None
    mean_odm_loss_c = 0
    mean_odm_loss_l = 0
    mean_arm_loss_c = 0
    mean_arm_loss_l = 0
    for iteration in range(start_iter, max_iter+10):
        if (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data.DataLoader(train_dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            if epoch % args.save_frequency == 0 and epoch > 0:
                torch.save(net.state_dict(), os.path.join(save_folder,args.version+'_'+args.dataset + '_epoches_'+
                           repr(epoch) + '.pth'))
            if epoch%args.test_frequency == 0 and epoch>0:
                net.eval()
                top_k = (300, 200)[args.dataset == 'COCO']
                if args.dataset == 'VOC':
                    APs,mAP = test_net(test_save_dir, net, detector, args.cuda, testset,
                             BaseTransform(net.module.size, rgb_means,rgb_std, (2, 0, 1)),
                             top_k, thresh=0.01)
                    APs = [str(num) for num in APs]
                    mAP = str(mAP)
                    log_file.write(str(iteration)+' APs:\n'+'\n'.join(APs))
                    log_file.write('mAP:\n'+mAP+'\n')
                else:
                    test_net(test_save_dir, net, detector, args.cuda, testset,
                                       BaseTransform(net.module.size, rgb_means,rgb_std, (2, 0, 1)),
                                       top_k, thresh=0.01)

                net.train()
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index  = stepvalues.index(iteration)+1
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                                    loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)


        # load train data
        images, targets = next(batch_iterator)

        #print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(),volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        out = net(images)
        arm_loc, arm_conf, odm_loc, odm_conf = out
        # backprop
        optimizer.zero_grad()
        #arm branch loss
        arm_loss_l,arm_loss_c = arm_criterion((arm_loc,arm_conf),priors,targets)
        #odm branch loss
        odm_loss_l, odm_loss_c = odm_criterion((odm_loc,odm_conf),priors,targets,(arm_loc,arm_conf),False)

        mean_arm_loss_c += arm_loss_c.data[0]
        mean_arm_loss_l += arm_loss_l.data[0]
        mean_odm_loss_c += odm_loss_c.data[0]
        mean_odm_loss_l += odm_loss_l.data[0]

        loss = arm_loss_l+arm_loss_c+odm_loss_l+odm_loss_c
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Total iter ' +
                  repr(iteration) + ' || AL: %.4f AC: %.4f OL: %.4f OC: %.4f||' % (
                mean_arm_loss_l/10,mean_arm_loss_c/10,mean_odm_loss_l/10,mean_odm_loss_c/10) +
                'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
            log_file.write('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Total iter ' +
                   repr(iteration) + ' || AL: %.4f AC: %.4f OL: %.4f OC: %.4f||' % (
                   mean_arm_loss_l/10,mean_arm_loss_c/10,mean_odm_loss_l/10,mean_odm_loss_c/10) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr)+'\n')

            mean_odm_loss_c = 0
            mean_odm_loss_l = 0
            mean_arm_loss_c = 0
            mean_arm_loss_l = 0
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())
    log_file.close()
    torch.save(net.state_dict(), os.path.join(save_folder ,
               'Final_' + args.version +'_' + args.dataset+ '.pth'))


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < args.warm_epoch:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * args.warm_epoch)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = (21, 81)[args.dataset == 'COCO']
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return


    for i in range(num_images):
        img = testset.pull_image(i)
        x = Variable(transform(img).unsqueeze(0),volatile=True)
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        out = net(x=x, test=True)  # forward pass
        arm_loc,arm_conf,odm_loc,odm_conf = out
        boxes, scores = detector.forward((odm_loc,odm_conf), priors,(arm_loc,arm_conf))
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores=scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            if args.dataset == 'VOC':
                cpu = False
            else:
                cpu = False

            keep = nms(c_dets, 0.45, force_cpu=cpu)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
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
        APs,mAP = testset.evaluate_detections(all_boxes, save_folder)
        return APs,mAP
    else:
        testset.evaluate_detections(all_boxes, save_folder)

if __name__ == '__main__':
    train()
