import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
GPU = False
if torch.cuda.is_available():
    GPU = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """


    def __init__(self, num_classes,overlap_thresh,prior_for_matching,bkg_label,neg_mining,neg_pos,neg_overlap,encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1,0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:,:-1].detach()
            labels = targets[idx][:,-1].detach()
            defaults = priors.detach()
            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)
        if GPU:
            loc_t = loc_t.to('cuda')
            conf_t = conf_t.to('cuda')

        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1,self.num_classes)
        #loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))
        loss_c = F.cross_entropy(batch_conf, conf_t.view(-1), ignore_index = -1, reduction='none')
        loss_c = loss_c.view(num, -1)

        # Hard Negative Mining
        pos_loss_c = loss_c[pos]
        loss_c[pos] = 0 # filter out pos boxes for now
        #loss_c = loss_c.view(num, -1)
        _,loss_idx = loss_c.sort(1, descending=True)
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1,keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg_loss_c = loss_c[neg]
        # Confidence Loss Including Positive and Negative Examples
        # pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        #conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        #targets_weighted = conf_t[(pos+neg).gt(0)]
        #loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        loss_c = pos_loss_c.sum() + neg_loss_c.sum()
        N = num_pos.data.sum().float()
        loss_l = loss_l/N
        loss_c = loss_c/N
        return loss_l, loss_c
class MultiBoxLoss2(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """


    def __init__(self, cfg, priors, use_gpu):
        super(MultiBoxLoss2, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.num_classes
        self.threshold = cfg.pos_th
        self.background_label = cfg.background_label
        self.negpos_ratio = cfg.negpos_ratio
        self.neg_overlap = cfg.neg_th
        self.variance = cfg.variance
        self.priors = priors

    def forward(self, predictions,  targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions
        priors = self.priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:,:-1].detach()
            labels = targets[idx][:,-1].detach()
            defaults = priors.detach()
            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)
        if GPU:
            loc_t = loc_t.to('cuda')
            conf_t = conf_t.to('cuda')

        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1,self.num_classes)
        #loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))
        loss_c = F.cross_entropy(batch_conf, conf_t.view(-1), ignore_index = -1, reduction='none')
        loss_c = loss_c.view(num, -1)

        # Hard Negative Mining
        pos_loss_c = loss_c[pos]
        loss_c[pos] = 0 # filter out pos boxes for now
        #loss_c = loss_c.view(num, -1)
        _,loss_idx = loss_c.sort(1, descending=True)
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1,keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg_loss_c = loss_c[neg]
        # Confidence Loss Including Positive and Negative Examples
        # pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        #conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        #targets_weighted = conf_t[(pos+neg).gt(0)]
        #loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        loss_c = pos_loss_c.sum() + neg_loss_c.sum()
        N = num_pos.data.sum().float()
        loss_l = loss_l/N
        loss_c = loss_c/N
        return loss_l, loss_c
