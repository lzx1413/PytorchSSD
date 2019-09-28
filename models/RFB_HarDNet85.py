import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)




class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1',ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2',DWConvLayer(out_channels, out_channels, stride=stride))

    def forward(self, x):
        return super().forward(x)

class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels,  stride=1,  bias=False):
        super().__init__()
        out_ch = out_channels

        groups = in_channels
        kernel = 3
        #print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')

        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                          stride=stride, padding=1, groups=groups, bias=bias))

        self.add_module('norm', nn.BatchNorm2d(groups))
    def forward(self, x):
        return super().forward(x)
        
class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=0, bias=False):
        super().__init__()
        self.out_channels = out_channels
        out_ch = out_channels
        groups = 1
        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        pad = kernel//2 if padding == 0 else padding
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel,
                                          stride=stride, padding=pad, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU(True))
    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          if dwconv:
            layers_.append(CombConvLayer(inch, outch))
          else:
            layers_.append(ConvLayer(inch, outch))

          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            x = torch.cat(tin, 1)
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out
        
        
class HarDNetBase(nn.Module):
    def __init__(self, depth_wise=False):
        super().__init__()
        first_ch  = [48, 96]
        second_kernel = 3

        ch_list = [  192, 256, 320, 480, 720]
        grmul = 1.7
        gr       = [  24, 24, 28, 36, 48]
        n_layers = [   8, 16, 16, 16, 16]
        
        if depth_wise:
          second_kernel = 1
          first_ch  = [24, 48]

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append (
             ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                       stride=2,  bias=False) )

        # Second Layer
        self.base.append ( ConvLayer(first_ch[0], first_ch[1],  kernel=second_kernel) )

        # Maxpooling or DWConv3x3 downsampling
        self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append ( blk )

            self.base.append ( ConvLayer(ch, ch_list[i], kernel=1) )
            ch = ch_list[i]
            if i == 0:
                self.base.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            elif i != blks-1 and i != 1 and i != 3:
                self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))











class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            # BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                      dilation=2 * visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        out = torch.cat((x0, x1), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        '''
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        '''
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class RFBNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, size,  extras, head, num_classes):
        super(RFBNet, self).__init__()
        self.num_classes = num_classes
        self.size = size
        
        self.base = HarDNetBase().base


        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # conv_4
        self.Norm = BasicRFB_a(320, 512, stride=1, scale=1.0)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax()

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(10):
            x = self.base[k](x)

        s = self.Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(10, len(self.base)):
            x = self.base[k](x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k % 2 == 0:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # print([o.size() for o in loc])

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

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def reload_weights(self):
        pretrained_path='weights/hardnet85_base.pth'
        self.base = HarDNetBase().base
        print(self.base)

        if pretrained_path is not None:
          weights = torch.load(pretrained_path)
          self.base.load_state_dict(weights)
          print("HarDNet68 Base Model has been reloaded.")


def add_extras(size, cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256 and size == 512:
                    layers += [BasicRFB(in_channels, cfg[k + 1], stride=2, scale=1.0, visual=1)]
                else:
                    layers += [BasicRFB(in_channels, cfg[k + 1], stride=2, scale=1.0, visual=2)]
            else:
                layers += [BasicRFB(in_channels, v, scale=1.0, visual=2)]
        in_channels = v
    if size == 512:
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=4, stride=1, padding=1)]
    elif size == 300:
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=1)]
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=1)]
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    return layers


extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256, 'S', 256],
}


def multibox(size, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [-2]
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(512,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(1024,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1024,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
    i = 1
    indicator = 0
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k % 2 == 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                      * num_classes, kernel_size=3, padding=1)]
            i += 1
    return  extra_layers, (loc_layers, conf_layers)


mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(size=300, num_classes=21):
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return RFBNet(size, *multibox(size, 
                                  add_extras(size, extras[str(size)], 720),
                                  mbox[str(size)], num_classes), num_classes=num_classes)
