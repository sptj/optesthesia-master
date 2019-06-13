import anchors as anchors
import torch as t
import numpy as np

from anchors import get_anchors_for_whole_fm, calc_iou, xyxy2cxcywh, cxcywh2xyxy

def encoder(boxes,labels,input_size):
    '''
    :param boxes:   note that boxes ordered by cxcywh
    :param labels:  boxes" labels
    :param input_size:
    :return:
    '''
    anchors_wfm=anchors.get_anchors_for_whole_fm()
    ious=anchors.calc_iou(t.Tensor(anchors_wfm),boxes)
    max_iou,max_idx=ious.max(1)
    boxes=boxes[max_idx]
    loc_targets_xy=(boxes[:,0:2]-t.Tensor(anchors_wfm)[:,0:2])/t.Tensor(anchors_wfm)[:,2:4]
    loc_targets_wh=t.log(boxes[:,2:4]/t.Tensor(anchors_wfm)[:,2:4])
    loc_targets=(t.cat([loc_targets_xy,loc_targets_wh],1))

    cls_targets=1+labels[max_idx]

    cls_targets[max_iou<0.5]=0
    cls_targets[(max_iou<0.5)&(max_iou>0.4)]=-1

    return loc_targets,cls_targets

def get_anchors_for_retinanet(origin_shape=np.array((1920, 1080))):
    # strides = np.array((8,16, 32, 64, 128))
    # anchor_edge_len = np.array((32, 64, 128, 256, 512))
    strides = np.array((8,16, 32))
    anchor_edge_len = np.array((32, 64,128))
    all_anchors = []
    for i, stride in enumerate(strides):
        fm_size = np.ceil(origin_shape / stride)
        all_anchors.append(get_anchors_for_whole_fm(origin_shape, fm_size, anchor_edge_len[i]))
    return np.vstack(all_anchors)


def encoder_for_retinanet(boxes, labels):
    anchors = get_anchors_for_retinanet(np.array((1080, 1080)))
    anchors1 = anchors.copy()
    anchors = t.from_numpy(anchors).double()
    anchors1 = t.from_numpy(anchors1).double()
    ious = calc_iou(cxcywh2xyxy(anchors1).double(), boxes.double())
    max_ious, max_ids = ious.max(1)
    boxes = (xyxy2cxcywh(boxes)).double()
    boxes = boxes[max_ids]
    loc_xy = (boxes[:, 0:2] - anchors[:, 0:2]) / anchors[:, 2:4]
    loc_wh = t.log(boxes[:, 2:4] / anchors[:, 2:4])
    loc_targets = t.cat((loc_xy, loc_wh), 1)
    cls_targets = 1 + labels[max_ids]
    cls_targets[max_ious < 0.5] = 0
    ignore = (max_ious < 0.5) & (max_ious > 0.4)
    cls_targets[ignore] = -1
    return cls_targets, loc_targets


def collate_fn_fake(batch):
    '''Pad images and encode targets.

    As for images are of different sizes, we need to pad them to the same size.

    Args:
      batch: (list) of images, cls_targets, loc_targets.

    Returns:
      padded images, stacked cls_targets, stacked loc_targets.
    '''
    imgs = [x[0] for x in batch]
    boxes = [x[1] for x in batch]
    labels = [x[2] for x in batch]

    num_imgs = len(imgs)
    inputs = t.zeros(num_imgs, 3, 1080, 1920)

    loc_targets = []
    cls_targets = []
    for i in range(num_imgs):
        inputs[i] = imgs[i]
        cls_target,loc_target  = encoder_for_retinanet(boxes[i], labels[i])
        loc_targets.append(loc_target)
        cls_targets.append(cls_target)
    return inputs,t.stack(cls_targets),  t.stack(loc_targets)

from encoder_ref import DataEncoder
def collate_fn(batch):
    '''Pad images and encode targets.

    As for images are of different sizes, we need to pad them to the same size.

    Args:
      batch: (list) of images, cls_targets, loc_targets.

    Returns:
      padded images, stacked cls_targets, stacked loc_targets.
    '''
    de=DataEncoder()
    imgs = [x[0] for x in batch]
    boxes = [x[1] for x in batch]
    labels = [x[2] for x in batch]

    num_imgs = len(imgs)
    inputs = t.zeros(num_imgs, 3, 1080, 1080)

    loc_targets = []
    cls_targets = []
    for i in range(num_imgs):
        inputs[i] = imgs[i]
        cls_target,loc_target  = de.encode(boxes[i], labels[i],(1080,1080))
        #cls_target, loc_target=encoder_for_retinanet(boxes[i], labels[i])

        # print("cls_target,loc_target",cls_target,loc_target )
        # print("cls_target_me, loc_target_me",cls_target_me, loc_target_me)
        # print("cls_target_error",t.max(cls_target.float()-cls_target_me.float()).item())
        # print("loc_target_error",t.max(loc_target.float()-loc_target_me.float()).item())
        loc_targets.append(loc_target)
        cls_targets.append(cls_target)
    return inputs,t.stack(cls_targets),  t.stack(loc_targets)





if __name__ =='__main__':
    a=t.Tensor([[0,0,20,20],[10,10,30,30]])
    b=t.Tensor([[0],[1]])
    b=t.Tensor([0,1])
    encoder(a,b,(60,60))




