import re
import math
#import pywt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from collections import OrderedDict
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

from IPython import embed

class Network:
    def construct(self, net, obj):
        targetClass = getattr(self, net)
        instance = targetClass(obj)
        return instance
    
    ###########################################################################
    ##############################      VGG      ##############################
    ###########################################################################
        
    class VGG(nn.Module):
        def __init__(self, obj, net_type, batch_norm):
            super(Network.VGG, self).__init__()
            
            self.features = self.make_layers(obj.input_ch, Network.cfg[net_type], batch_norm=batch_norm)
            
            num_strides = sum([layer == 'M' for layer in Network.cfg[net_type]])
            kernel_numel = int((obj.padded_im_size / (2**num_strides))**2)

            relu1 = nn.ReLU(inplace=False)
            relu2 = nn.ReLU(inplace=False)
            
            lin1 = nn.Linear(512 * kernel_numel, 4096, bias=False)
            lin2 = nn.Linear(4096, 4096, bias=False)
            lin3 = nn.Linear(4096, 1000)
            
            bn1 = nn.BatchNorm1d(4096)
            bn2 = nn.BatchNorm1d(4096)
            
            self.classifier = nn.Sequential(
                lin1,
                bn1,
                relu1,
                lin2,
                bn2,
                relu2,
                lin3
            )
            
            self._initialize_weights()
            
            mod = list(self.classifier.children())
            mod.pop()
            
            lin4 = torch.nn.Linear(4096, obj.num_classes)
            
            mod.append(lin4)
            self.classifier = torch.nn.Sequential(*mod)
            self.classifier[-1].weight.data.normal_(0, 0.01)
            self.classifier[-1].bias.data.zero_()
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    try:
                        m.bias.data.zero_()
                    except:
                        pass
                    
        def make_layers(self, input_ch, cfg, batch_norm=False):
            layers  = []

            in_channels = input_ch
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    relu = nn.ReLU(inplace=False)

                    if batch_norm:
                        bn = nn.BatchNorm2d(v)

                        layers += [conv2d, bn, relu]
                    else:
                        layers += [conv2d, relu]
                    in_channels = v
            return nn.Sequential(*layers)
    
    
    cfg = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    
    class VGG11(VGG):
        def __init__(self, obj):
            super(Network.VGG11, self).__init__(obj, 'A', False)
            
    class VGG11_bn(VGG):
        def __init__(self, obj):
            super(Network.VGG11_bn, self).__init__(obj, 'A', True)

    class VGG13(VGG):
        def __init__(self, obj):
            super(Network.VGG13, self).__init__(obj, 'B', False)
            
    class VGG13_bn(VGG):
        def __init__(self, obj):
            super(Network.VGG13_bn, self).__init__(obj, 'B', True)
        
    class VGG16(VGG):
        def __init__(self, obj):
            super(Network.VGG16, self).__init__(obj, 'D', False)
            
    class VGG16_bn(VGG):
        def __init__(self, obj):
            super(Network.VGG16_bn, self).__init__(obj, 'D', True)

    class VGG19(VGG):
        def __init__(self, obj):
            super(Network.VGG19, self).__init__(obj, 'E', False)
            
    class VGG19_bn(VGG):
        def __init__(self, obj):
            super(Network.VGG19_bn, self).__init__(obj, 'E', True)
    
    ###########################################################################
    #############################      ResNet      ############################
    ###########################################################################
    
    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)
    
    @staticmethod
    def conv1x1(in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
    
    class BasicBlock(nn.Module):
        expansion = 1
    
        def __init__(self, inplanes, planes, last, stride=1, downsample=None):
            super(Network.BasicBlock, self).__init__()
            self.conv1 = Network.conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU(inplace=True)
            
            self.conv2 = Network.conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            if last:
                self.relu2 = nn.ReLU(inplace=False)
            else:
                self.relu2 = nn.ReLU(inplace=True)
            
            self.downsample = downsample
            self.stride = stride
    
        def forward(self, x):
            residual = x
    
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
    
            out = self.conv2(out)
            out = self.bn2(out)
    
            if self.downsample is not None:
                residual = self.downsample(x)
    
            out += residual
            out = self.relu2(out)
    
            return out
    
    
    class Bottleneck(nn.Module):
        expansion = 4
    
        def __init__(self, inplanes, planes, last, stride=1, downsample=None):
            super(Network.Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
            if last:
                self.relu3 = nn.ReLU(inplace=False)
            else:
                self.relu3 = nn.ReLU(inplace=True)
                
            self.downsample = downsample
            self.stride = stride
    
        def forward(self, x):
            residual = x
    
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
    
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)
    
            out = self.conv3(out)
            out = self.bn3(out)
    
            if self.downsample is not None:
                residual = self.downsample(x)
    
            out += residual
            out = self.relu3(out)
    
            return out
    
    class ResNet(nn.Module):
    
        def __init__(self, obj, block, layers):
            self.obj = obj
            self.inplanes = 64
            super(Network.ResNet, self).__init__()
            
            if obj.resnet_type == 'big':
                self.conv1 = nn.Conv2d(obj.input_ch, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
            elif obj.resnet_type == 'small':
                self.conv1 = Network.conv3x3(obj.input_ch, 64)
                
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU(inplace=True)
            
            if obj.resnet_type == 'big':
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            
            if obj.resnet_type == 'big':
                num_strides = 5
            elif obj.resnet_type == 'small':
                num_strides = 3
            
            kernel_sz = int(obj.padded_im_size / (2**num_strides))
            self.avgpool = nn.AvgPool2d(kernel_sz, stride=1)
            self.fc = nn.Linear(512 * block.expansion, obj.num_classes)
            
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
    
            layers = []
            layers.append(block(self.inplanes, planes, False, stride, downsample))
            
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                
                last = (planes == 512) & (i == blocks-1)
                
                cur_layers = block(self.inplanes, planes, last)                
                layers.append(cur_layers)
    
            return nn.Sequential(*layers)
        
        def forward(self, x):
                        
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            
            if hasattr(self, 'maxpool'):
                x = self.maxpool(x)
    
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
    
            return x
            
    class ResNet18(ResNet):
        def __init__(self, obj):
            super(Network.ResNet18, self).__init__(obj, Network.BasicBlock, [2, 2, 2, 2])
    
    class ResNet34(ResNet):
        def __init__(self, obj):
            super(Network.ResNet34, self).__init__(obj, Network.BasicBlock, [3, 4, 6, 3])
    
    class ResNet50(ResNet):
        def __init__(self, obj):
            super(Network.ResNet50, self).__init__(obj, Network.Bottleneck, [3, 4, 6, 3])
    
    class ResNet101(ResNet):
        def __init__(self, obj):
            super(Network.ResNet101, self).__init__(obj, Network.Bottleneck, [3, 4, 23, 3])

    class ResNet152(ResNet):
        def __init__(self, obj):
            super(Network.ResNet152, self).__init__(obj, Network.Bottleneck, [3, 8, 36, 3])

    ###########################################################################
    ###########################      DenseNet      ############################
    ###########################################################################

    class DenseNetBasicBlock(nn.Module):
        def __init__(self, in_planes, out_planes, dropRate=0.0):
            super(Network.DenseNetBasicBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.droprate = dropRate
            
        def forward(self, x):
            out = self.conv1(self.relu(self.bn1(x)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            return torch.cat([x, out], 1)
    
    class BottleneckBlock(nn.Module):
        def __init__(self, in_planes, out_planes, dropRate=0.0):
            super(Network.BottleneckBlock, self).__init__()
            inter_planes = out_planes * 4
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(inter_planes)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.droprate = dropRate
            
        def forward(self, x):
            out = self.conv1(self.relu1(self.bn1(x)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
            out = self.conv2(self.relu2(self.bn2(out)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
            return torch.cat([x, out], 1)
    
    class TransitionBlock(nn.Module):
        def __init__(self, in_planes, out_planes, dropRate=0.0):
            super(Network.TransitionBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
            self.droprate = dropRate
            
        def forward(self, x):
            out = self.conv1(self.relu(self.bn1(x)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
            return F.avg_pool2d(out, 2)
        
    class DenseBlock(nn.Module):
        def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
            super(Network.DenseBlock, self).__init__()
            self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
            
        def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
            layers = []
            self.relus = []
            self.convs = []
            for i in range(nb_layers):
                b = block(in_planes+i*growth_rate, growth_rate, dropRate)
                layers.append(b)
            return nn.Sequential(*layers)
        
        def forward(self, x):
            return self.layer(x)
    
    class DenseNet3(nn.Module):
        def __init__(self, obj, depth, growth_rate=12,
                     reduction=0.5, bottleneck=True, dropRate=0.0):
            super(Network.DenseNet3, self).__init__()
            in_planes = 2 * growth_rate
            n = (depth - 4) / 3
            if bottleneck == True:
                n = n/2
                block = Network.BottleneckBlock
            else:
                block = Network.DenseNetBasicBlock
            n = int(n)
            
            # 1st conv before any dense block
            self.conv1 = nn.Conv2d(obj.input_ch, in_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            
            # 1st block
            self.block1 = Network.DenseBlock(n, in_planes, growth_rate, block, dropRate)
            in_planes = int(in_planes+n*growth_rate)
            self.trans1 = Network.TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
            in_planes = int(math.floor(in_planes*reduction))
            
            # 2nd block
            self.block2 = Network.DenseBlock(n, in_planes, growth_rate, block, dropRate)
            in_planes = int(in_planes+n*growth_rate)
            self.trans2 = Network.TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
            in_planes = int(math.floor(in_planes*reduction))
            
            # 3rd block
            self.block3 = Network.DenseBlock(n, in_planes, growth_rate, block, dropRate)
            in_planes = int(in_planes+n*growth_rate)
            
            # global average pooling and classifier
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(in_planes, obj.num_classes)
            self.in_planes = in_planes
            
            kernel_sz = int(obj.padded_im_size / (2**2))    
            self.avgpool = nn.AvgPool2d(kernel_size=kernel_sz, stride=1)
    
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()
                    
        def forward(self, x):
            out = self.conv1(x)
            out = self.trans1(self.block1(out))
            out = self.trans2(self.block2(out))
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = self.avgpool(out)
            out = out.view(-1, self.in_planes)
            return self.fc(out)
        
    class DenseNet3_40(DenseNet3):
        def __init__(self, obj):
            super(Network.DenseNet3_40, self).__init__(obj, depth=40, growth_rate=12, reduction=1, bottleneck=False, dropRate=0.0)
            
            