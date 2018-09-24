import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .VGG16 import get_vgg16_fms
from layers.modules.l2norm import L2Norm


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=True, up_size=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = F.interpolate(x, size=(self.up_size, self.up_size), mode='bilinear', align_corners=True)
        return x

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(fea_channels, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    assert len(fea_channels) == len(cfg)
    for i, fea_channel in enumerate(fea_channels):
        loc_layers += [nn.Conv2d(fea_channel, cfg[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(fea_channel, cfg[i] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


extras = {
    '300': [256, 512, 128, 'S', 256],
    '512': [256, 512, 128, 'S', 256],
}
mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}
fea_channels = {
    '300': [512, 512, 256, 256, 256, 256],
    '512': [512, 512, 256, 256, 256, 256, 256]}


class RefineDet(nn.Module):

    def __init__(self, num_classes, size, use_refine=True):

        super(RefineDet, self).__init__()
        self.num_classes = num_classes
        self.size = size
        # SSD network
        #self.base =  get_vgg16_fms(fm_ids=[22,29,34], pool5=True, conv6_dilation=3)
        self.base =  get_vgg16_fms(fm_ids=[22,29,34], pool5=True)
        self.use_refine = use_refine
        self.L2Norm_4_3 = L2Norm(512, 10)
        self.L2Norm_5_3 = L2Norm(512, 8)
        self.last_layer_trans = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                              nn.ReLU(inplace=True))

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

        self.trans_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)), \
                                           nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)), \
                                           nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
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

    def get_pyramid_feature(self, x):
        arm_fms = self.base(x)
        arm_fms[0] = self.L2Norm_4_3(arm_fms[0])
        arm_fms[1] = self.L2Norm_5_3(arm_fms[1])
        x = arm_fms[-1]
        x = self.extras(x)
        arm_fms.append(x)
        odm_fms = list()
        odm_fms.append(self.last_layer_trans(x))
        arm_fms.reverse()
        x = odm_fms[0]
        for (arm_fm, t, u, l) in zip(arm_fms[1:], self.trans_layers, self.up_layers, self.latent_layrs):
            x = F.relu(l(F.relu(t(arm_fm) + u(x))))
            odm_fms.append(x)
        arm_fms.reverse()
        odm_fms.reverse()
        return arm_fms, odm_fms



    def forward(self, x, test=False):
        loc = list()
        conf = list()

        arm_fms, odm_fms = self.get_pyramid_feature(x)

        # apply multibox head to source layers
        arm_loc_list = list()
        arm_conf_list = list()
        odm_loc_list = list()
        odm_conf_list = list()
        if self.use_refine:
            for (x, l, c) in zip(arm_fms, self.arm_loc, self.arm_conf):
                arm_loc_list.append(l(x).permute(0, 2, 3, 1).contiguous())
                arm_conf_list.append(c(x).permute(0, 2, 3, 1).contiguous())
            arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc_list], 1)
            arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf_list], 1)

        for (x, l, c) in zip(odm_fms, self.odm_loc, self.odm_conf):
            odm_loc_list.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf_list.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc_list], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf_list], 1)

        # apply multibox head to source layers

        if test:
            if self.use_refine:
                output = (
                    arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(arm_conf.view(-1, 2)),  # conf preds
                    odm_loc.view(odm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(odm_conf.view(-1, self.num_classes)),  # conf preds
                )
            else:
                output = (
                    odm_loc.view(odm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(odm_conf.view(-1, self.num_classes)),  # conf preds
                )
        else:
            if self.use_refine:
                output = (
                    arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                    arm_conf.view(arm_conf.size(0), -1, 2),  # conf preds
                    odm_loc.view(odm_loc.size(0), -1, 4),  # loc preds
                    odm_conf.view(odm_conf.size(0), -1, self.num_classes),  # conf preds
                )
            else:
                output = (
                    odm_loc.view(odm_loc.size(0), -1, 4),  # loc preds
                    odm_conf.view(odm_conf.size(0), -1, self.num_classes),  # conf preds
                )

        return output

    def init_model(self, base_model_path):

        base_weights = torch.load(base_model_path)
        print('Loading base network...')
        self.base.layers.load_state_dict(base_weights)


        def xavier(param):
            init.xavier_uniform(param)


        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.xavier_normal_(m.state_dict()[key])
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0
        print('Initializing weights...')
        self.extras.apply(weights_init)
        self.last_layer_trans.apply(weights_init)
        self.trans_layers.apply(weights_init)
        self.odm_loc.apply(weights_init)
        self.odm_conf.apply(weights_init)
        self.latent_layrs.apply(weights_init)
        self.up_layers.apply(weights_init)
        if self.use_refine:
            self.arm_loc.apply(weights_init)
            self.arm_conf.apply(weights_init)


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')




def build_net(size=300, num_classes=21, use_refine = True):
    if size != 320 and size != 512:
        return

    return RefineDet(num_classes=num_classes,size=size, use_refine = use_refine)
