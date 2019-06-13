from dataset import VocDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from encoder import collate_fn
from RetinaNet import retinanet
from loss_myself import focal_loss
from torch.optim import SGD
import torch
import os


# from torch.utils.tensorboard import SummaryWriter

def train_model(epoch):
    net.train()
    sum_loss = 0
    for index, (imgs, cls_target, loc_target) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs = imgs.cuda(0)
        cls_pred, loc_pred = net(imgs)
        loss = criterion(
            loc_preds=loc_pred,
            loc_targets=loc_target,
            cls_preds=cls_pred,
            cls_targets=cls_target)
        sum_loss = sum_loss + loss.item()
        print("status->train,epoch->{},batch_idx->{:04},train_loss->{:4.8f},avg_loss->{:4.8f}".format(
            epoch,
            index,
            loss.item(),
            sum_loss / (index + 1)
        ))
        loss.backward()
        optimizer.step()
        # writer.add_scalar('loss',loss.item())


def valid_model(epoch):
    net.eval()
    sum_loss = 0
    i = 0
    for index, (imgs, cls_target, loc_target) in enumerate(valid_loader):
        optimizer.zero_grad()
        imgs = imgs.cuda(0)
        cls_pred, loc_pred = net(imgs)
        loss = criterion(
            loc_preds=loc_pred,
            loc_targets=loc_target,
            cls_preds=cls_pred,
            cls_targets=cls_target)
        if torch.isinf(loss):
            print('loss inf item is ', index)
            print(cls_target)
            print("loss is inf")
        else:
            i = i + 1
            sum_loss = sum_loss + loss.item()
            print("status->valid,epoch->{},batch_idx->{:04},train_loss->{:>4.8f},avg_loss->{:>4.8f}".format(
                epoch,
                index,
                loss.item(),
                sum_loss / i
            ))

    global best_loss
    if sum_loss < best_loss:
        best_loss = sum_loss
        state = {
            'net': net.state_dict(),
            'loss': best_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')


if __name__ == '__main__':
    img_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_set = VocDataset(
        root=r"H:\drone_image_and_annotation_mixed\train",
        train=True,
        transform=img_trans,
        input_size=1080
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=12,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        timeout=0,
        worker_init_fn=None)
    valid_set = VocDataset(
        root=r"H:\drone_image_and_annotation_mixed\test",
        train=True,
        transform=img_trans,
        input_size=1080
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        timeout=0,
        worker_init_fn=None)
    criterion = focal_loss(num_cls=1)
    best_loss = float('inf')
    net = retinanet(num_classes=1)
    if os.path.isfile(r"./checkpoint/ckpt.pth"):
        print("resume from ckpt...")
        param = torch.load(r"./checkpoint/ckpt.pth")
        net.load_state_dict(param['net'])
        best_loss=param['loss']
    else:
        print("start train from random...")
    optimizer = SGD(net.parameters(), lr=3e-4, momentum=0.9, weight_decay=1e-4)
    net.cuda(0)
    # writer = SummaryWriter()
    start_epoch = 0
    for epoch in range(start_epoch, start_epoch + 200):
        train_model(epoch)
        valid_model(epoch)
