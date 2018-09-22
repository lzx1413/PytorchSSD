import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .VGG16 import get_vgg16_fms


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


def feature_transform_module(size):
    if size ==  300:
        up_size = 38
    elif size == 512:
        up_size = 64

    layers = []
    # conv4_3
    layers += [BasicConv(512, 256, kernel_size=1, padding=0)]
    # fc_7
    layers += [BasicConv(1024, 256, kernel_size=1, padding=0, up_size=up_size)]
    layers += [BasicConv(256, 256, kernel_size=1, padding=0, up_size=up_size)]
    return layers


def pyramid_feature_extractor(size):
    if size == 300:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0),
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0)]
    elif size == 512:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=4, padding=1, stride=1)]
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


class FSSD(nn.Module):

    def __init__(self, num_classes, size):

        super(FSSD, self).__init__()
        self.num_classes = num_classes
        self.size = size
        # SSD network
        self.base =  get_vgg16_fms()
        self.extras = nn.ModuleList(add_extras(extras[str(self.size)], 1024))
        self.ft_module = nn.ModuleList(feature_transform_module(self.size))
        self.pyramid_ext = nn.ModuleList(pyramid_feature_extractor(self.size))
        self.fea_bn = nn.BatchNorm2d(256 * len(self.ft_module), affine=True)

        head = multibox(fea_channels[str(self.size)], mbox[str(self.size)], self.num_classes)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax()

    def get_pyramid_feature(self, x):
        source_fms = list()
        source_fms += self.base(x)
        x = source_fms[-1]
        for f in self.extras:
            x = F.relu(f(x), inplace=True)
        source_fms.append(x)
        assert len(source_fms) == len(self.ft_module)
        transformed_features = list()
        for k, v in enumerate(self.ft_module):
            x = v(source_fms[k])
            transformed_features.append(x)
        concat_fea = torch.cat(transformed_features, 1)
        x = self.fea_bn(concat_fea)

        pyramid_fea = list()


        for k,v in enumerate(self.pyramid_ext):
            x = v(x)
            pyramid_fea.append(x)
        return pyramid_fea



    def forward(self, x, test=False):
        loc = list()
        conf = list()

        pyramid_fea = self.get_pyramid_feature(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(pyramid_fea, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if test:
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
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
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0
        print('Initializing weights...')
        self.extras.apply(weights_init)
        self.ft_module.apply(weights_init)
        self.pyramid_ext.apply(weights_init)
        self.loc.apply(weights_init)
        self.conf.apply(weights_init)


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')




def build_net(size=300, num_classes=21):
    if size != 300 and size != 512:
        print("Error: Sorry only FSSD300 and FSSD512 is supported currently!")
        return

    return FSSD(num_classes=num_classes,size=size)
