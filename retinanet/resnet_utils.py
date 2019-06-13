import torch as t
import torch.nn as nn


class simple_conv(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride=(1, 1)):
        super(simple_conv, self).__init__()
        if isinstance(kernel_size, tuple):
            padding0 = int(kernel_size[0] / 2)
            padding1 = int(kernel_size[1] / 2)
            padding = (padding0, padding1)
        elif isinstance(kernel_size, int):
            padding = int(kernel_size / 2)
        else:
            raise Exception('unknow kernel_size type')
        self.conv = nn.Conv2d(in_channels=in_plane, out_channels=out_plane, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_plane)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class basic_block(nn.Module):
    expansion = 1

    def __init__(self, in_plane, out_plane, stride):
        super(basic_block, self).__init__()
        self.residual = nn.Sequential(
            simple_conv(in_plane, out_plane, kernel_size=(3, 3), stride=stride),
            nn.ReLU(inplace=True),
            simple_conv(out_plane, out_plane, kernel_size=(3, 3), stride=1)
        )
        if in_plane != out_plane or stride != 1:
            self.projection = simple_conv(in_plane, out_plane, kernel_size=(1, 1), stride=stride)
        else:
            self.projection = nn.Sequential()

    def forward(self, x):
        res = self.residual(x)
        prj = self.projection(x)
        out = nn.functional.relu_(res + prj)
        return out


class bottle_neck(nn.Module):
    expansion = 4

    def __init__(self, in_plane, out_plane, stride):
        super(bottle_neck, self).__init__()
        self.residual=nn.Sequential(
            simple_conv(in_plane,out_plane,kernel_size=(1,1),stride=1),
            nn.ReLU(inplace=True),
            simple_conv(out_plane,out_plane,kernel_size=(3,3),stride=stride),
            nn.ReLU(inplace=True),
            simple_conv(out_plane,out_plane*self.expansion,kernel_size=(1,1),stride=1),
        )

        if in_plane != out_plane*self.expansion or stride != 1:
            self.projection = simple_conv(in_plane, out_plane*self.expansion, kernel_size=(1, 1), stride=stride)
        else:
            self.projection = nn.Sequential()
    def forward(self,x):
        res=self.residual(x)
        prj=self.projection(x)
        out=nn.functional.relu_(res+prj)
        return out

class resnet_core(nn.Module):
    def __init__(self, block, block_stacked, origin_pic_channels=3):
        super(resnet_core, self).__init__()
        self.in_plane = 64
        self.layer0 = simple_conv(origin_pic_channels, 64, kernel_size=(7, 7), stride=(2, 2))
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_stacked[0], stride=1)
        self.layer2 = self._make_layer(block, 128, block_stacked[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_stacked[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_stacked[3], stride=2)
        self.layer5 = nn.AvgPool2d(kernel_size=(7, 7))
        self.fc = nn.Linear(512*block.expansion, 1000)

    def _make_layer(self, block, out_plane, num_of_block_stacked, stride):
        strides=num_of_block_stacked*[1]
        strides[0]=stride
        layers=[]
        for spec_stide in strides:
            layers.append(block(self.in_plane,out_plane,spec_stide))
            self.in_plane=out_plane*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x=nn.functional.relu_(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1)
        x = self.fc(x)
        return x

def self_resnet18():
    return resnet_core(basic_block,[2,2,2,2])
def self_resnet34():
    return resnet_core(basic_block, [3,4,6,3])
def self_resnet50():
    return resnet_core(bottle_neck, [3,4,6,3])

if __name__ == '__main__':
    test_image = t.rand((1, 3, 224, 224))
    # ref_parms = t.load(r'C:\Users\sptj\PycharmProjects\optesthesia\retinanet\resnet18-5c106cde.pth')
    ref_parms = t.load(r'C:\Users\sptj\PycharmProjects\optesthesia\retinanet\resnet50-19c8e357.pth')
    net_spt = self_resnet50()
    state_spt=net_spt.state_dict()
    import numpy as np

    k_ref = list(ref_parms.keys())
    # print(*k_ref,sep='\n')
    k_ref.pop(-1)
    k_ref.pop(-1)

    k_spt = list(state_spt.keys())
    # print(*k_spt,sep="\n")
    k_spt.pop(-1)
    k_spt.pop(-1)

    # for (ref), (spt) in zip(k_ref,k_spt):
    #     print('{:<30}{:<30}'.format((ref), (spt)))
    k_ref = np.array(k_ref).reshape(-1, 5).tolist()
    k_spt = np.array(k_spt).reshape(-1, 5).tolist()
    from collections import OrderedDict

    translate_table = OrderedDict()
    for ref, spt in zip(k_ref, k_spt):
        translate_table[spt[0]] = ref[0]
        translate_table[spt[1]] = ref[3]
        translate_table[spt[2]] = ref[4]
        translate_table[spt[3]] = ref[1]
        translate_table[spt[4]] = ref[2]
    translate_table['fc.weight'] = 'fc.weight'
    translate_table['fc.bias'] = 'fc.bias'

    # for k,v in translate_table.items():
    #     print('{:<40}{:<40}'.format((k), (v)))


    from torchvision.models import resnet50
    net_ref = resnet50(pretrained=True)

    for i in state_spt.keys():
        state_spt[i] = ref_parms[translate_table[i]]

    net_spt.load_state_dict(state_spt)
    result_sptj_trained = net_spt(test_image)
    result_ref = net_ref(test_image)

    print('{:<20}{:<20}'.format('ref', 'spt'))

    print(result_ref.view(-1).argmax())
    print(result_sptj_trained.view(-1).argmax())
    a=result_ref.view(-1).tolist()
    b=result_sptj_trained.view(-1).tolist()

    i = 0
    for ref, spt in zip(a, b):
        print('{:<30}{:<30}'.format((ref), (spt)))
        i += 1
        if i == 10:
            break
    print((result_ref-result_sptj_trained).tolist())

    t.save(net_spt.state_dict(),'self_resnet50.pth')