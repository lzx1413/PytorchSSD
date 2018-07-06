import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from .base_models import vgg, vgg_base


def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


vgg_base = {
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


class RefineSSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, size, num_classes, use_refine=False):
        super(RefineSSD, self).__init__()
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.size = size
        self.use_refine = use_refine

        # SSD network
        self.base = nn.ModuleList(vgg(vgg_base['320'], 3))
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm_4_3 = L2Norm(512, 10)
        self.L2Norm_5_3 = L2Norm(512, 8)
        self.last_layer_trans = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.extras = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True), \
                                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))

        if use_refine:
            self.arm_loc = nn.ModuleList([nn.Conv2d(512, 12, kernel_size=3, stride=1, padding=1), \
                                          nn.Conv2d(512, 12, kernel_size=3, stride=1, padding=1), \
                                          nn.Conv2d(1024, 12, kernel_size=3, stride=1, padding=1), \
                                          nn.Conv2d(512, 12, kernel_size=3, stride=1, padding=1), \
                                          ])
            self.arm_conf = nn.ModuleList([nn.Conv2d(512, 6, kernel_size=3, stride=1, padding=1), \
                                           nn.Conv2d(512, 6, kernel_size=3, stride=1, padding=1), \
                                           nn.Conv2d(1024, 6, kernel_size=3, stride=1, padding=1), \
                                           nn.Conv2d(512, 6, kernel_size=3, stride=1, padding=1), \
                                           ])
        self.odm_loc = nn.ModuleList([nn.Conv2d(256, 12, kernel_size=3, stride=1, padding=1), \
                                      nn.Conv2d(256, 12, kernel_size=3, stride=1, padding=1), \
                                      nn.Conv2d(256, 12, kernel_size=3, stride=1, padding=1), \
                                      nn.Conv2d(256, 12, kernel_size=3, stride=1, padding=1), \
                                      ])
        self.odm_conf = nn.ModuleList([nn.Conv2d(256, 3*num_classes, kernel_size=3, stride=1, padding=1), \
                                       nn.Conv2d(256, 3*num_classes, kernel_size=3, stride=1, padding=1), \
                                       nn.Conv2d(256, 3*num_classes, kernel_size=3, stride=1, padding=1), \
                                       nn.Conv2d(256, 3*num_classes, kernel_size=3, stride=1, padding=1), \
                                       ])
        self.trans_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)), \
                                           nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)), \
                                           nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)), \
                                           ])
        self.up_layers = nn.ModuleList([nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0), ])
        self.latent_layrs = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                           ])

        self.softmax = nn.Softmax()

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        arm_sources = list()
        arm_loc_list = list()
        arm_conf_list = list()
        obm_loc_list = list()
        obm_conf_list = list()
        obm_sources = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        s = self.L2Norm_4_3(x)
        arm_sources.append(s)

        # apply vgg up to conv5_3
        for k in range(23, 30):
            x = self.base[k](x)
        s = self.L2Norm_5_3(x)
        arm_sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.base)):
            x = self.base[k](x)
        arm_sources.append(x)
        # conv6_2
        x = self.extras(x)
        arm_sources.append(x)
        # apply multibox head to arm branch
        if self.use_refine:
            for (x, l, c) in zip(arm_sources, self.arm_loc, self.arm_conf):
                arm_loc_list.append(l(x).permute(0, 2, 3, 1).contiguous())
                arm_conf_list.append(c(x).permute(0, 2, 3, 1).contiguous())
            arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc_list], 1)
            arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf_list], 1)
        x = self.last_layer_trans(x)
        obm_sources.append(x)

        # get transformed layers
        trans_layer_list = list()
        for (x_t, t) in zip(arm_sources, self.trans_layers):
            trans_layer_list.append(t(x_t))
        # fpn module
        trans_layer_list.reverse()
        arm_sources.reverse()
        for (t, u, l) in zip(trans_layer_list, self.up_layers, self.latent_layrs):
            x = F.relu(l(F.relu(u(x) + t, inplace=True)), inplace=True)
            obm_sources.append(x)
        obm_sources.reverse()
        for (x, l, c) in zip(obm_sources, self.odm_loc, self.odm_conf):
            obm_loc_list.append(l(x).permute(0, 2, 3, 1).contiguous())
            obm_conf_list.append(c(x).permute(0, 2, 3, 1).contiguous())
        obm_loc = torch.cat([o.view(o.size(0), -1) for o in obm_loc_list], 1)
        obm_conf = torch.cat([o.view(o.size(0), -1) for o in obm_conf_list], 1)

        # apply multibox head to source layers

        if test:
            if self.use_refine:
                output = (
                    arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(arm_conf.view(-1, 2)),  # conf preds
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(obm_conf.view(-1, self.num_classes)),  # conf preds
                )
            else:
                output = (
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(obm_conf.view(-1, self.num_classes)),  # conf preds
                )
        else:
            if self.use_refine:
                output = (
                    arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                    arm_conf.view(arm_conf.size(0), -1, 2),  # conf preds
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    obm_conf.view(obm_conf.size(0), -1, self.num_classes),  # conf preds
                )
            else:
                output = (
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    obm_conf.view(obm_conf.size(0), -1, self.num_classes),  # conf preds
                )

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_net(size=320, num_classes=21, use_refine=False):
    if size != 320:
        print("Error: Sorry only SSD300 and SSD512 is supported currently!")
        return

    return RefineSSD(size, num_classes=num_classes, use_refine=use_refine)
