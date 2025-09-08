import math
import torch
import torch.nn as nn
from torch.nn import init


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class Chanel_Compress(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Chanel_Compress,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_feat, math.floor(in_feat * (0.9))),
            nn.BatchNorm1d(math.floor(in_feat * (0.9))),
            nn.ReLU(),
            nn.Linear(math.floor(in_feat * (0.9)), math.floor(in_feat * (0.85))),
            nn.BatchNorm1d(math.floor(in_feat * (0.85))),
            nn.ReLU(),
            nn.Linear(math.floor(in_feat * (0.85)), out_feat)
        )

    def forward(self,x):
        x = x.reshape(-1,x.size(2))
        x = self.model(x)
        return x

class VIB(nn.Module):
    def __init__(self, in_feat, out_feat, num_class):
        super(VIB,self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bottleneck = Chanel_Compress(in_feat=self.in_feat,out_feat=self.out_feat)

        self.classifier = nn.Sequential(
            nn.Linear(self.out_feat, math.floor(self.out_feat * (0.9))),
            nn.BatchNorm1d(math.floor(self.out_feat * (0.9))),
            nn.ReLU(),
            nn.Linear(math.floor(self.out_feat * (0.9)), math.floor(self.out_feat * (0.85))),
            nn.BatchNorm1d(math.floor(self.out_feat * (0.85))),
            nn.ReLU(),
            nn.Linear(math.floor(self.out_feat * (0.85)), num_class)
        )
        self.classifier.apply(weights_init_classifier)
    def forward(self,v):
        z_given_v = self.bottleneck(v)
        p_y_given_z = self.classifier(z_given_v)
        return p_y_given_z, z_given_v

if __name__=="__main__":
    ib = VIB(in_feat=64,out_feat=32,num_class=12095)
    input = torch.randn(size=(32, 199, 64))
    out = ib(input)
    print(out)
    print(0)


