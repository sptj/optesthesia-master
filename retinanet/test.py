from dataset import VocDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from encoder import collate_fn
from RetinaNet import retinanet
from loss_myself import focal_loss
from torch.optim import SGD
import torch
import os
from PIL import Image, ImageDraw
from encoder_ref import DataEncoder
net = retinanet(num_classes=1)
if os.path.isfile(r"./checkpoint/ckpt.pth"):
    print("resume from ckpt...")
    param=torch.load(r"./checkpoint/ckpt.pth")
    net.load_state_dict(param['net'])
    #best_loss=param['loss']
else:
    print("start train from random...")
net.cuda()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])




def predict_img(img):
    img_src=img.copy()
    img=transform(img)
    img=img.cuda()
    img=img.unsqueeze(0)
    cls_preds ,loc_preds= net(img)
    loc_preds=loc_preds.data.squeeze().cpu()
    cls_preds=cls_preds.data.squeeze().cpu()
    encoder = DataEncoder()

    boxes,labels  = encoder.decode(loc_preds, cls_preds, (1920, 1080))
    draw = ImageDraw.Draw(img_src)
    for box in boxes:
        draw.rectangle(list(box), outline='red')
    return img_src

if __name__ == '__main__':
    test1=Image.open(r"C:\Users\sptj\Pictures\example\video201871_1217_frame_1.jpg")
    result=predict_img(test1)
    result.save('./result.png')






