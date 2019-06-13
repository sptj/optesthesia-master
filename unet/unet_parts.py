import torch as t
import torch.nn as nn
from torch.autograd import Variable

class double_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(double_conv,self).__init__()
        self.layer0=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        x=self.layer0(x)
        return x
class down(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(down,self).__init__()
        self.layer0=nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch,out_ch)
        )
    def forward(self, x):
        x=self.layer0(x)
        return x

class up(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(up,self).__init__()
        self.layer0=nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch),
            double_conv(out_ch,out_ch),
        )













if __name__ == '__main__':
    test_img=t.ones((2,3,572,572))
    # print(test_img)
    net=double_conv(3,6)
    print(net(test_img))

