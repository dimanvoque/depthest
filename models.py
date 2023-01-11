"""
Configures all the 5 models' architectures 
"""

import os
import torch
import torch.nn as nn
import torchvision.models
import collections
import math
import torch.nn.functional as F
import imagenet.mobilenetv3 as imagenet



"""
Constructs necessary layers 
"""

class Identity(nn.Module):
    # a dummy identity module
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, stride=2):
        super(Unpool, self).__init__()

        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.mask = torch.zeros(1, 1, stride, stride)
        self.mask[:,:,0,0] = 1

    def forward(self, x):
        assert x.dim() == 4
        num_channels = x.size(1)
        return F.conv_transpose2d(x,
            self.mask.detach().type_as(x).expand(num_channels, 1, -1, -1),
            stride=self.stride, groups=num_channels)

def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv(in_channels, out_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=padding,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )

def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )

def convt(in_channels, out_channels, kernel_size):
    stride = 2
    padding = (kernel_size - 1) // 2
    output_padding = kernel_size % 2
    assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,
                stride,padding,output_padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

def convt_dw(channels, kernel_size):
    stride = 2
    padding = (kernel_size - 1) // 2
    output_padding = kernel_size % 2
    assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"
    return nn.Sequential(
            nn.ConvTranspose2d(channels,channels,kernel_size,
                stride,padding,output_padding,bias=False,groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

def upconv(in_channels, out_channels):
    return nn.Sequential(
        Unpool(2),
        nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

class upproj(nn.Module):
    # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
    #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
    #   bottom branch: 5*5 conv -> batchnorm

    def __init__(self, in_channels, out_channels):
        super(upproj, self).__init__()
        self.unpool = Unpool(2)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.unpool(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return F.relu(x1 + x2)

class Decoder(nn.Module):
    names = ['deconv{}{}'.format(i,dw) for i in range(3,10,2) for dw in ['', 'dw']]
    names.append("deconv2") # default decoder
    names.append("upconv")
    names.append("upproj")
    for i in range(3,10,2):
        for dw in ['', 'dw']:
            names.append("nnconv{}{}".format(i, dw))
            names.append("blconv{}{}".format(i, dw))
            names.append("shuffle{}{}".format(i, dw))

class DeConv(nn.Module):

    def __init__(self, kernel_size, dw):
        super(DeConv, self).__init__()
        if dw:
            self.convt1 = nn.Sequential(
                convt_dw(1024, kernel_size),
                pointwise(1024, 512))
            self.convt2 = nn.Sequential(
                convt_dw(512, kernel_size),
                pointwise(512, 256))
            self.convt3 = nn.Sequential(
                convt_dw(256, kernel_size),
                pointwise(256, 128))
            self.convt4 = nn.Sequential(
                convt_dw(128, kernel_size),
                pointwise(128, 64))
            self.convt5 = nn.Sequential(
                convt_dw(64, kernel_size),
                pointwise(64, 32))
        else:
            self.convt1 = convt(1024, 512, kernel_size)
            self.convt2 = convt(512, 256, kernel_size)
            self.convt3 = convt(256, 128, kernel_size)
            self.convt4 = convt(128, 64, kernel_size)
            self.convt5 = convt(64, 32, kernel_size)
        self.convf = pointwise(32, 1)

    def forward(self, x):
        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)
        x = self.convt4(x)
        x = self.convt5(x)
        x = self.convf(x)
        return x


class UpConv(nn.Module):

    def __init__(self):
        super(UpConv, self).__init__()
        self.upconv1 = upconv(1024, 512)
        self.upconv2 = upconv(512, 256)
        self.upconv3 = upconv(256, 128)
        self.upconv4 = upconv(128, 64)
        self.upconv5 = upconv(64, 32)
        self.convf = pointwise(32, 1)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        x = self.convf(x)
        return x

class UpProj(nn.Module):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    def __init__(self):
        super(UpProj, self).__init__()
        self.upproj1 = upproj(1024, 512)
        self.upproj2 = upproj(512, 256)
        self.upproj3 = upproj(256, 128)
        self.upproj4 = upproj(128, 64)
        self.upproj5 = upproj(64, 32)
        self.convf = pointwise(32, 1)

    def forward(self, x):
        x = self.upproj1(x)
        x = self.upproj2(x)
        x = self.upproj3(x)
        x = self.upproj4(x)
        x = self.upproj5(x)
        x = self.convf(x)
        return x

class NNConv(nn.Module):

    def __init__(self, kernel_size, dw):
        super(NNConv, self).__init__()
        if dw:
            self.conv1 = nn.Sequential(
                depthwise(960, kernel_size),
                pointwise(960, 480))
            self.conv2 = nn.Sequential(
                depthwise(480, kernel_size),
                pointwise(480, 240))
            self.conv3 = nn.Sequential(
                depthwise(240, kernel_size),
                pointwise(240, 120))
            self.conv4 = nn.Sequential(
                depthwise(120, kernel_size),
                pointwise(120, 60))
            self.conv5 = nn.Sequential(
                depthwise(60, kernel_size),
                pointwise(60, 30))
            self.conv6 = pointwise(30, 1)
        else:
            self.conv1 = conv(960, 480, kernel_size)
            self.conv2 = conv(480, 240, kernel_size)
            self.conv3 = conv(240, 120, kernel_size)
            self.conv4 = conv(120, 60, kernel_size)
            self.conv5 = conv(60, 30, kernel_size)
            self.conv6 = pointwise(30, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv5(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv6(x)
        return x

class BLConv(NNConv):

    def __init__(self, kernel_size, dw):
        super(BLConv, self).__init__(kernel_size, dw)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv5(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv6(x)
        return x

class ShuffleConv(nn.Module):

    def __init__(self, kernel_size, dw):
        super(ShuffleConv, self).__init__()
        if dw:
            self.conv1 = nn.Sequential(
                depthwise(256, kernel_size),
                pointwise(256, 256))
            self.conv2 = nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 64))
            self.conv3 = nn.Sequential(
                depthwise(16, kernel_size),
                pointwise(16, 16))
            self.conv4 = nn.Sequential(
                depthwise(4, kernel_size),
                pointwise(4, 4))
        else:
            self.conv1 = conv(256, 256, kernel_size)
            self.conv2 = conv(64, 64, kernel_size)
            self.conv3 = conv(16, 16, kernel_size)
            self.conv4 = conv(4, 4, kernel_size)

    def forward(self, x):
        x = F.pixel_shuffle(x, 2)
        x = self.conv1(x)

        x = F.pixel_shuffle(x, 2)
        x = self.conv2(x)

        x = F.pixel_shuffle(x, 2)
        x = self.conv3(x)

        x = F.pixel_shuffle(x, 2)
        x = self.conv4(x)

        x = F.pixel_shuffle(x, 2)
        return x

def choose_decoder(decoder):
    depthwise = ('dw' in decoder)
    if decoder[:6] == 'deconv':
        assert len(decoder)==7 or (len(decoder)==9 and 'dw' in decoder)
        kernel_size = int(decoder[6])
        model = DeConv(kernel_size, depthwise)
    elif decoder == "upproj":
        model = UpProj()
    elif decoder == "upconv":
        model = UpConv()
    elif decoder[:7] == 'shuffle':
        assert len(decoder)==8 or (len(decoder)==10 and 'dw' in decoder)
        kernel_size = int(decoder[7])
        model = ShuffleConv(kernel_size, depthwise)
    elif decoder[:6] == 'nnconv':
        assert len(decoder)==7 or (len(decoder)==9 and 'dw' in decoder)
        kernel_size = int(decoder[6])
        model = NNConv(kernel_size, depthwise)
    elif decoder[:6] == 'blconv':
        assert len(decoder)==7 or (len(decoder)==9 and 'dw' in decoder)
        kernel_size = int(decoder[6])
        model = BLConv(kernel_size, depthwise)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)
    model.apply(weights_init)
    return model


class MobileNetV3L_NNConv5GU(nn.Module):   #defines MobileNetV3L-NNConv5GU architecture
    def __init__(self, output_size, in_channels=3, pretrained=True):

        super(MobileNetV3L_NNConv5GU, self).__init__()
        self.output_size = output_size
        self.mobilenetv3 = imagenet.mobilenetv3_large()   #gets the large encoder
        if pretrained:   #if we need pretrained encoder downloads it
            self.mobilenetv3.load_state_dict(torch.load('imagenet/pretrained/mobilenetv3-large-1cd25616.pth'))
        else:  #if we don't need pretrained encoder initialize the weights
            self.mobilenetv3.apply(weights_init)

        """
        Cut last layers built for classification 
        """

        childs = list(self.mobilenetv3.children())

        if in_channels == 3:
            self.mobilenetv3 = nn.Sequential(*(childs[i] for i in range(2)))
        else:
            def conv_bn(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU6(inplace=True)
                )

            self.mobilenetv3 = nn.Sequential(
                conv_bn(in_channels,  32, 2),
                *(childs[i] for i in range(2)))

        """
        Define decoder 
        """
        
        kernel_size = 5   #define the size of decoder's kernel size 
        # self.decode_conv1 = conv(960, 480, kernel_size)
        # self.decode_conv2 = conv(480, 240, kernel_size)
        # self.decode_conv3 = conv(240, 120, kernel_size)
        # self.decode_conv4 = conv(120, 60, kernel_size)
        # self.decode_conv5 = conv(60, 30, kernel_size)
        self.decode_conv1 = nn.Sequential(
            depthwise(960, kernel_size),
            pointwise(960, 480))
        self.decode_conv2 = nn.Sequential(
            depthwise(480, kernel_size),
            pointwise(480, 240))
        self.decode_conv3 = nn.Sequential(
            depthwise(240, kernel_size),
            pointwise(240, 120))
        self.decode_conv4 = nn.Sequential(
            depthwise(120, kernel_size),
            pointwise(120, 60))
        self.decode_conv5 = nn.Sequential(
            depthwise(60, kernel_size),
            pointwise(60, 30))
        self.decode_conv6 = pointwise(30, 1)

        """
        Initialize the weights of decoder
        """
        
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        x = self.mobilenetv3(x)
        x = self.decode_conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.decode_conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.decode_conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.decode_conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.decode_conv5(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.decode_conv6(x)

        return x


class MobileNetV3SkipAddL_NNConv5R(nn.Module):   #defines MobileNetV3SkipAddL-NNConv5R architecture
    def __init__(self, output_size, pretrained=True):

        super(MobileNetV3SkipAddL_NNConv5R, self).__init__()
        self.output_size = output_size
        self.mobilenetv3 = imagenet.mobilenetv3_large()   #gets the large encoder
        if pretrained:   #if we need pretrained encoder downloads it
            self.mobilenetv3.load_state_dict(torch.load('imagenet/pretrained/mobilenetv3-large-1cd25616.pth'))
        else:   #if we don't need pretrained encoder initialize the weights
            self.mobilenetv3.apply(weights_init)

        """
        Cut last layers built for classification 
        """

        childs = list(self.mobilenetv3.children())
        self.mobilenetv3 = nn.Sequential(*(childs[i] for i in range(2)))
        childs12 = torch.nn.Sequential(*childs[0], *childs[1])


        for i in range (len(childs12)):
            setattr( self, 'conv{}'.format(i), childs12[i])

        """
        Define decoder 
        """

        kernel_size = 5
        # self.decode_conv1 = conv(960, 80, kernel_size)
        # self.decode_conv2 = conv(80, 40, kernel_size)
        # self.decode_conv3 = conv(40, 24, kernel_size)
        # self.decode_conv4 = conv(24, 16, kernel_size)
        # self.decode_conv5 = conv(16, 3, kernel_size)
        self.decode_conv1 = nn.Sequential(
            depthwise(960, kernel_size),
            pointwise(960, 80))
        self.decode_conv2 = nn.Sequential(
            depthwise(80, kernel_size),
            pointwise(80, 40))
        self.decode_conv3 = nn.Sequential(
            depthwise(40, kernel_size),
            pointwise(40, 24))
        self.decode_conv4 = nn.Sequential(
            depthwise(24, kernel_size),
            pointwise(24, 16))
        self.decode_conv5 = nn.Sequential(
            depthwise(16, kernel_size),
            pointwise(16, 3))
        self.decode_conv6 = pointwise(3, 1)

        """
        Initialize the weights of decoder
        """
        
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc3
        # dec 2: enc6

        childs = list(self.mobilenetv3.children())
        self.mobilenetv3 = nn.Sequential(*(childs[i] for i in range(2)))

        childs12 = torch.nn.Sequential(*childs[0], *childs[1])

        """
        Build skip-connections and forward propagation 
        """

        for i in range (len(childs12)):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            #print("{}: {}".format(i, x.size()))
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==6:
                x3 = x
        for i in range(1,6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i==4:
                x = x + x1
            elif i==3:
                x = x + x2
            elif i==2:
                x = x + x3
            #print("{}: {}".format(i, x.size()))
        x = self.decode_conv6(x)
        return x

class MobileNetV3SkipAddL_NNConv5S(nn.Module):   #defines MobileNetV3SkipAddL-NNConv5S architecture
    def __init__(self, output_size, pretrained=True):

        super(MobileNetV3SkipAddL_NNConv5S, self).__init__()
        self.output_size = output_size
        self.mobilenetv3 = imagenet.mobilenetv3_large()   #gets the large encoder
        if pretrained:   #if we need pretrained encoder downloads it
            self.mobilenetv3.load_state_dict(torch.load('imagenet/pretrained/mobilenetv3-large-1cd25616.pth'))
        else:   #if we don't need pretrained encoder initialize the weights
            self.mobilenetv3.apply(weights_init)

        """
        Cut last layers built for classification 
        """

        childs = list(self.mobilenetv3.children())
        self.mobilenetv3 = nn.Sequential(*(childs[i] for i in range(2)))
        childs12 = torch.nn.Sequential(*childs[0], *childs[1])


        for i in range (len(childs12)):
            setattr( self, 'conv{}'.format(i), childs12[i])

        """
        Define decoder 
        """

        kernel_size = 5
        # self.decode_conv1 = conv(960, 80, kernel_size)
        # self.decode_conv2 = conv(80, 40, kernel_size)
        # self.decode_conv3 = conv(40, 24, kernel_size)
        # self.decode_conv4 = conv(24, 16, kernel_size)
        # self.decode_conv5 = conv(16, 1, kernel_size)
        self.decode_conv1 = nn.Sequential(
            depthwise(960, kernel_size),
            pointwise(960, 80))
        self.decode_conv2 = nn.Sequential(
            depthwise(80, kernel_size),
            pointwise(80, 40))
        self.decode_conv3 = nn.Sequential(
            depthwise(40, kernel_size),
            pointwise(40, 24))
        self.decode_conv4 = nn.Sequential(
            depthwise(24, kernel_size),
            pointwise(24, 16))
        self.decode_conv5 = nn.Sequential(
            depthwise(16, kernel_size),
            pointwise(16, 1))

        """
        Initialize the weights of decoder
        """
        
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc3
        # dec 2: enc6

        childs = list(self.mobilenetv3.children())
        self.mobilenetv3 = nn.Sequential(*(childs[i] for i in range(2)))

        childs12 = torch.nn.Sequential(*childs[0], *childs[1])

        """
        Build skip-connections and forward propagation 
        """

        for i in range (len(childs12)):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            #print("{}: {}".format(i, x.size()))
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==6:
                x3 = x
        for i in range(1,6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i==4:
                x = x + x1
            elif i==3:
                x = x + x2
            elif i==2:
                x = x + x3
            #print("{}: {}".format(i, x.size()))
        #x = self.decode_conv6(x)
        return x


class MobileNetV3SkipAddS_NNConv5R(nn.Module):
    def __init__(self, output_size, pretrained=True):

        super(MobileNetV3SkipAddS_NNConv5R, self).__init__()
        self.output_size = output_size
        self.mobilenetv3 = imagenet.mobilenetv3_small()   #gets the small encoder
        if pretrained:   #if we need pretrained encoder downloads it
            self.mobilenetv3.load_state_dict(torch.load('imagenet/pretrained/mobilenetv3-small-55df8e1f.pth'))
        else:   #if we don't need pretrained encoder initialize the weights
            self.mobilenetv3.apply(weights_init)

        """
        Cut last layers built for classification 
        """

        childs = list(self.mobilenetv3.children())
        self.mobilenetv3 = nn.Sequential(*(childs[i] for i in range(2)))
        childs12 = torch.nn.Sequential(*childs[0], *childs[1])

        for i in range(len(childs12)):
            setattr(self, 'conv{}'.format(i), childs12[i])

        """
        Define decoder 
        """

        kernel_size = 5
        # self.decode_conv1 = conv(576, 40, kernel_size)
        # self.decode_conv2 = conv(40, 24, kernel_size)
        # self.decode_conv3 = conv(24, 16, kernel_size)
        # self.decode_conv4 = conv(16, 3, kernel_size)
        # self.decode_conv5 = conv(3, 1, kernel_size)
        self.decode_conv1 = nn.Sequential(
            depthwise(576, kernel_size),
            pointwise(576, 40))
        self.decode_conv2 = nn.Sequential(
            depthwise(40, kernel_size),
            pointwise(40, 24))
        self.decode_conv3 = nn.Sequential(
            depthwise(24, kernel_size),
            pointwise(24, 16))
        self.decode_conv4 = nn.Sequential(
            depthwise(16, kernel_size),
            pointwise(16, 3))
        self.decode_conv5 = pointwise(3, 1)

        """
        Initialize the weights of decoder
        """

        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)

    def forward(self, x):
        # skip connections: dec3: enc1
        # dec 2: enc3

        childs = list(self.mobilenetv3.children())
        self.mobilenetv3 = nn.Sequential(*(childs[i] for i in range(2)))

        childs12 = torch.nn.Sequential(*childs[0], *childs[1])

        """
        Build skip-connections and forward propagation 
        """

        for i in range(len(childs12)):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            # print("{}: {}".format(i, x.size()))
            if i == 1:
                x2 = x
            elif i == 3:
                x3 = x
        for i in range(1, 6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i == 3:
                x = x + x2
            elif i == 2:
                x = x + x3
            # print("{}: {}".format(i, x.size()))
        return x

class MobileNetV3S_NNConv5GU(nn.Module):   #defines MobileNetV3S-NNConv5GU architecture
    def __init__(self, output_size, in_channels=3, pretrained=True):

        super(MobileNetV3S_NNConv5GU, self).__init__()
        self.output_size = output_size
        self.mobilenetv3 = imagenet.mobilenetv3_small()   #gets the small encoder
        if pretrained:   #if we need pretrained encoder downloads it
            self.mobilenetv3.load_state_dict(torch.load('imagenet/pretrained/mobilenetv3-small-55df8e1f.pth'))
        else:   #if we don't need pretrained encoder initialize the weights
            self.mobilenetv3.apply(weights_init)

        """
        Cut last layers built for classification 
        """

        childs = list(self.mobilenetv3.children())

        if in_channels == 3:
            self.mobilenetv3 = nn.Sequential(*(childs[i] for i in range(2)))
        else:
            def conv_bn(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU6(inplace=True)
                )

            self.mobilenetv3 = nn.Sequential(
                conv_bn(in_channels,  32, 2),
                *(childs[i] for i in range(2)))

        """
        Define decoder 
        """
        
        kernel_size = 5
        # self.decode_conv1 = conv(576, 288, kernel_size)
        # self.decode_conv2 = conv(288, 144, kernel_size)
        # self.decode_conv3 = conv(144, 72, kernel_size)
        # self.decode_conv4 = conv(72, 36, kernel_size)
        # self.decode_conv5 = conv(36, 1, kernel_size)

        self.decode_conv1 = nn.Sequential(
            depthwise(576, kernel_size),
            pointwise(576, 288))
        self.decode_conv2 = nn.Sequential(
            depthwise(288, kernel_size),
            pointwise(288, 144))
        self.decode_conv3 = nn.Sequential(
            depthwise(144, kernel_size),
            pointwise(144, 72))
        self.decode_conv4 = nn.Sequential(
            depthwise(72, kernel_size),
            pointwise(72, 36))
        self.decode_conv5 = nn.Sequential(
            depthwise(36, kernel_size),
            pointwise(36, 1))

        """
        Initialize the weights of decoder
        """

        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)

    def forward(self, x):
        x = self.mobilenetv3(x)
        x = self.decode_conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.decode_conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.decode_conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.decode_conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.decode_conv5(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        return x
