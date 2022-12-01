import torch
import torch.nn.functional as F
from torch import nn
import itertools
from collections import OrderedDict
import Interp
from torchvision.models import vgg16
import numpy as np
import csv

class I2INetRDN(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, n_downsample=3, skip=True, dim=32, n_res=8, norm='in', activ='relu', pad_type='reflect', denormalize=False, final_activ='tanh'):
        super(I2INetRDN, self).__init__()

        self.skip=skip
        self.denormalize = denormalize

        # project to feature space
        self.conv_in = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)

        # downsampling blocks
        self.down_blocks = nn.ModuleList()
        for i in range(n_downsample):
            self.down_blocks.append( Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type) )
            dim *= 2

        # residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(n_res):
            self.res_blocks.append(RDB(dim,dim, 2) )

        # upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_downsample):
            self.up_blocks.append( nn.Sequential(nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)) )
            dim //= 2

        # project to image space
        self.conv_out = Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=final_activ, pad_type=pad_type)

        #self.apply(weights_init('kaiming'))
        #self.apply(weights_init('gaussian'))


    def forward(self, x):
        # normalize image and save mean/var if using denormalization
        if self.denormalize:
            x_mean = x.view(x.size(0), x.size(1), -1).mean(2).view(x.size(0), x.size(1), 1, 1)
            x_var = x.view(x.size(0), x.size(1), -1).var(2).view(x.size(0), x.size(1), 1, 1)
            x = (x-x_mean)/x_var

        # project to feature space
        x = self.conv_in(x)

        # downsampling blocks
        xs = []
        for block in self.down_blocks:
            xs += [x]
            x = block(x)

        # residual blocks
        for block in self.res_blocks:
            x = block(x)

        # upsampling blocks
        for block, skip in zip(self.up_blocks, reversed(xs)):
            x = block(x)
            if self.skip:
                x = x + skip

        # project to image space
        x = self.conv_out(x)

        # denormalize if necessary
        if self.denormalize:
            x = x*x_var+x_mean
        return x




class CCNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, layers=5, dim=32, norm='gn', activ='relu', pad_type='reflect', final_activ='tanh'):
        super(CCNet, self).__init__()
        self.model = []
        #self.model += [Conv2dBlock(input_dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(input_dim, dim, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(layers-2):
            self.model += [Conv2dBlock(dim, dim, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, output_dim, 1, 1, 0, norm='none', activation=final_activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
    def forward(self, x):
        return self.model(x)

# class vgg_features(nn.Module):
#     def __init__(self):
#         super(vgg_features, self).__init__()
#         # get vgg16 features up to conv 4_3
#         self.model_vgg = nn.Sequential(*list(vgg16(pretrained=True).features)[:23])
#         # will not need to compute gradients
#         for param in self.parameters():
#             param.requires_grad=False
#
#     def forward(self, x, renormalize=True):
#         # change normaliztion form [-1,1] to VGG normalization
#         if renormalize:
#             x = ((x*.5+.5)-torch.cuda.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))/torch.cuda.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1)
#         return self.model_vgg(x)

class vgg_features(nn.Module):
    def __init__(self):
        super(vgg_features, self).__init__()
        # get vgg16 features up to conv 4_3
        self.model = nn.Sequential(*list(vgg16(pretrained=True).features)[:23])
        #self.model = self.model.cuda(1)
        self.model = self.model.cuda()
        # will not need to compute gradients
        for param in self.parameters():
            param.requires_grad=False

    def forward(self, x):


        
        x = ((x*.5+.5)-torch.cuda.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))/torch.cuda.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1)
            #x = ((x*.5+.5)-torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1))/torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        #x = x.cpu()
        #x = x.cuda(1)
        #x = self.model(x)
        #x = x.cuda()
        #x = x.cpu()
        #x = x.cuda(0)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, gan_type='lsgan', input_dim=3, dim=64, n_layers=4, norm='bn', activ='lrelu', pad_type='reflect'):
        super(Discriminator, self).__init__()
        self.gan_type = gan_type
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 4, 2, 1, norm='none', activation=activ, pad_type=pad_type)]
        for i in range(n_layers - 1):
            self.model += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [nn.Conv2d(dim, 1, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        #self.apply(weights_init('gaussian'))

    def forward(self, input):
        return self.model(input).mean(3).mean(2).squeeze()


##################################################################################
# Basic Blocks
##################################################################################
class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
	    nn.BatchNorm2d(G),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)
        self.CA = CALayer(G0)

    def forward(self, x):
        return self.CA(self.LFF(self.convs(x))) + x

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', transposed=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(norm_dim/8, norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transposed:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x




##################################################################################
# weight initialization
##################################################################################

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                nn.init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant(m.bias.data, 0.0)
    return init_fun
