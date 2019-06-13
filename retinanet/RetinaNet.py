import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        return t.cat(cls_results, 1), t.cat(loc_results, 1)




if __name__ == '__main__':
    # box=t.Tensor([[ 370.6299,  366.7027,   39.1409,   20.7683]])
    # encoder_for_retinanet(cxcywh2xyxy(box),t.Tensor([0]))
    model=retinanet(1)
    model.cuda()
    example = t.rand(4, 3, 1920, 1080).cuda()
    result=model(example)
    print(result[0].size(),result[1].size())

# if __name__ == '__main1__':
#     import cv2
#     from torchvision import transforms
#     from format_converter.image_format_transfer import trans_from_cv2_to_PIL
#     anchors=get_anchors_for_retinanet()
#     img=cv2.imread(r'D:\drone_image_and_annotation_mixed\test\video2018620_1579_frame_261.jpg')
#     orig_img=img.copy()
#     img=trans_from_cv2_to_PIL(img)
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ])
#     img=transform(img)
#     net = retinanet(num_classes=1)
#     spt_parms = t.load(r'C:\Users\sptj\PycharmProjects\optesthesia\retinanet\self_retina.pth')
#     net.load_state_dict(spt_parms)
#     net.cuda(0)
#     img=img.unsqueeze(dim=0)
#     img=img.cuda(0)
#     a, b = net(img)
#     target=anchors[a.argmax()]
#     cv2.rectangle(orig_img,target[0:2],target[2:4])
#     cv2.imshow(orig_img)
#     cv2.waitKey()
#
#
