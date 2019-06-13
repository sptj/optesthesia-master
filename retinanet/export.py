from dataset import VocDataset
from torchvision.transforms import transforms
import torch
import torch as t
import os
import torch.nn as nn
from PIL import Image, ImageDraw
from encoder_ref import DataEncoder
from ShuffleNetV2 import shufflenetv2_x1_5
class retinanet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes):
        super(retinanet, self).__init__()
        self.num_classes = num_classes
        self.fpn =shufflenetv2_x1_5()
        self.loc_pred = self._make_head(self.num_anchors * 4)
        self.cls_pred = self._make_head(self.num_anchors * num_classes)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(
                nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)))
            layers.append(nn.ReLU(True))
        layers.append(
            nn.Conv2d(in_channels=192, out_channels=out_planes, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)))
        return nn.Sequential(*layers)

    def forward(self, x):
        fms = self.fpn(x)
        loc_results = []
        cls_results = []
        for fm in fms:
            loc_result = self.loc_pred(fm)
            cls_result = self.cls_pred(fm)
            loc_result = loc_result.permute(0, 2, 3, 1).contiguous().view(loc_result.shape[0], -1, 4)
            cls_result = cls_result.permute(0, 2, 3, 1).contiguous().view(cls_result.shape[0], -1, self.num_classes)
            loc_results.append(loc_result)
            cls_results.append(cls_result)
        return t.cat(cls_results, 1)
net = retinanet(num_classes=1)
if os.path.isfile(r"./checkpoint/ckpt.pth"):
    print("resume from ckpt...")
    param=torch.load(r"./checkpoint/ckpt.pth")
    net.load_state_dict(param['net'])
    #best_loss=param['loss']
else:
    print("start train from random...")
net.cuda()
net.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 1920, 1080).cuda()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(net, example)
traced_script_module.save("shuffle-1043.pt")
#
#
# # read image
image = Image.open('result.png').convert('RGB')

image=image.resize((1920,1080))
img=image.copy()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
print(image.size)
image = transform(image)

# forward
loc_preds= traced_script_module(image.unsqueeze(0).cuda())
loc_preds=loc_preds.argmax()
print(loc_preds)
encoder = DataEncoder()
ref_table = encoder._get_anchor_boxes(torch.Tensor([1920, 1080]))

boxes = [ref_table[loc_preds]]
box=boxes[0]
print(boxes)
# box[0] = (box[0] - box[2]/2)
# box[1] = (box[1] - box[3]/2)
# box[2] = (box[2] + box[0])
# box[3] = (box[3] + box[1])
#
# print(ref_table[215999])
#
# draw = ImageDraw.Draw(img)
#
# draw.rectangle(list(box), outline='red')
# img.show()
