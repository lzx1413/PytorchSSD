import torch
from torch.autograd import Function

from utils.box_utils import decode, center_size


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, cfg, object_score=0):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.object_score = object_score
        # self.thresh = thresh

        # Parameters used in nms.
        self.variance = cfg['variance']

    def forward(self, predictions, prior, arm_data=None):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        loc, conf = predictions
        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size
        if arm_data:
            arm_loc, arm_conf = arm_data
            arm_loc_data = arm_loc.data
            arm_conf_data = arm_conf.data
            arm_object_conf = arm_conf_data[:, 1:]
            no_object_index = arm_object_conf <= self.object_score
            conf_data[no_object_index.expand_as(conf_data)] = 0

        self.num_priors = prior_data.size(0)
        self.boxes = torch.zeros(num, self.num_priors, 4)
        self.scores = torch.zeros(num, self.num_priors, self.num_classes)

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)

        else:
            conf_preds = conf_data.view(num, self.num_priors,
                                        self.num_classes)
            self.boxes.expand(num, self.num_priors, 4)
            self.scores.expand(num, self.num_priors, self.num_classes)
        # Decode predictions into bboxes.
        for i in range(num):
            if arm_data:
                default = decode(arm_loc_data[i], prior_data, self.variance)
                default = center_size(default)
            else:
                default = prior_data
            decoded_boxes = decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            '''
            c_mask = conf_scores.gt(self.thresh)
            decoded_boxes = decoded_boxes[c_mask]
            conf_scores = conf_scores[c_mask]
            '''

            self.boxes[i] = decoded_boxes
            self.scores[i] = conf_scores

        return self.boxes, self.scores
