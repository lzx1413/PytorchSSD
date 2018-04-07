import sys

import os
import torch
import torch.nn as nn

sys.path.append('./')
from .mobilenet import mobilenet_1


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
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x


class FSSD(nn.Module):
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

    def __init__(self, size, head, ft_module, pyramid_ext, num_classes):
        super(FSSD, self).__init__()
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.size = size

        # SSD network
        self.base = mobilenet_1()
        # Layer learns to scale the l2 normalized features from conv4_3
        self.ft_module = nn.ModuleList(ft_module)
        self.pyramid_ext = nn.ModuleList(pyramid_ext)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.fea_bn = nn.BatchNorm2d(256 * len(self.ft_module), affine=True)

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
        source_features = list()
        transformed_features = list()
        loc = list()
        conf = list()

        base_out = self.base(x)
        source_features.append(base_out[0])  # mobilenet 4_1
        source_features.append(base_out[1])  # mobilent_5_5
        source_features.append(base_out[2])  # mobilenet 6_1

        assert len(self.ft_module) == len(source_features)
        for k, v in enumerate(self.ft_module):
            transformed_features.append(v(source_features[k]))
        concat_fea = torch.cat(transformed_features, 1)
        x = self.fea_bn(concat_fea)
        fea_bn = x
        pyramid_fea = list()
        for k, v in enumerate(self.pyramid_ext):
            x = v(x)
            pyramid_fea.append(x)
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
            features = ()
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
            features = (
                fea_bn
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            state_dict = torch.load(base_file, map_location=lambda storage, loc: storage)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                head = k[:7]
                if head == 'module.':
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            self.base.load_state_dict(new_state_dict)
            print('Finished!')

        else:
            print('Sorry only .pth and .pkl files supported.')


def feature_transform_module(scale_factor):
    layers = []
    # conv4_1
    layers += [BasicConv(int(256 * scale_factor), 256, kernel_size=1, padding=0)]
    # conv5_5
    layers += [BasicConv(int(512 * scale_factor), 256, kernel_size=1, padding=0, up_size=38)]
    # conv6_mpo1
    layers += [BasicConv(int(1024 * scale_factor), 256, kernel_size=1, padding=0, up_size=38)]
    return layers


def pyramid_feature_extractor():
    '''
    layers = [BasicConv(256*3,512,kernel_size=3,stride=1,padding=1),BasicConv(512,512,kernel_size=3,stride=2,padding=1), \
              BasicConv(512,256,kernel_size=3,stride=2,padding=1),BasicConv(256,256,kernel_size=3,stride=2,padding=1), \
              BasicConv(256,256,kernel_size=3,stride=1,padding=0),BasicConv(256,256,kernel_size=3,stride=1,padding=0)]
    '''
    from .mobilenet import DepthWiseBlock
    layers = [DepthWiseBlock(256 * 3, 512, stride=1), DepthWiseBlock(512, 512, stride=2),
              DepthWiseBlock(512, 256, stride=2), DepthWiseBlock(256, 256, stride=2), \
              DepthWiseBlock(256, 128, stride=1, padding=0), DepthWiseBlock(128, 128, stride=1, padding=0)]

    return layers


def multibox(fea_channels, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    assert len(fea_channels) == len(cfg)
    for i, fea_channel in enumerate(fea_channels):
        loc_layers += [nn.Conv2d(fea_channel, cfg[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(fea_channel, cfg[i] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
}
fea_channels = [512, 512, 256, 256, 128, 128]


def build_net(size=300, num_classes=21):
    if size != 300 and size != 512:
        print("Error: Sorry only SSD300 and SSD512 is supported currently!")
        return

    return FSSD(size, multibox(fea_channels, mbox[str(size)], num_classes), feature_transform_module(1),
                pyramid_feature_extractor(), \
                num_classes=num_classes)


net = build_net()
print(net)
