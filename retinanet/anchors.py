import torch as t


def cxcywh2xyxy(cxcywh):
    cxcywh[:, 0:2] = cxcywh[:, 0:2] - cxcywh[:, 2:4] / 2
    cxcywh[:, 2:4] = cxcywh[:, 0:2] + cxcywh[:, 2:4] - 1
    return cxcywh


def xyxy2cxcywh(xyxy):
    xyxy[:, 2:4] = xyxy[:, 2:4] - xyxy[:, 0:2] + 1
    xyxy[:, 0:2] = xyxy[:, 0:2] + xyxy[:, 2:4] / 2
    return xyxy


def calc_iou(grid_boxes, target_boxes, order='xyxy'):
    if (order == 'cxcywh'):
        grid_boxes = cxcywh2xyxy(grid_boxes)
        target_boxes = cxcywh2xyxy(target_boxes)
    lt = t.max(grid_boxes[:, None, 0:2], target_boxes[:, 0:2])
    rb = t.min(grid_boxes[:, None, 2:4], target_boxes[:, 2:4])
    wh = (rb - lt + 1)
    wh = t.clamp(wh, min=0)
    inter = wh.prod(dim=2)
    grid_boxes_area = (grid_boxes[:, None, 2:4] - grid_boxes[:, None, 0:2] + 1).prod(dim=2)
    target_boxes_area = (target_boxes[:, 2:4] - target_boxes[:, 0:2] + 1).prod(dim=1)

    return inter / (grid_boxes_area + target_boxes_area - inter)


def test_calc_iou():
    grid_boxes = t.Tensor([
        [0, 0, 20, 20],
        [0, 20, 20, 20],
        [20, 0, 20, 20],
        [20, 20, 20, 20],
    ])
    target_boxes = t.Tensor(
        [[10, 10, 20, 20],
         [20, 20, 10, 10]]
    )
    print(calc_iou(grid_boxes, target_boxes, 'cxcywh'))


if __name__ == '__main__':
    (test_calc_iou())


def test_cxcywh2xyxy():
    cxcywh = t.Tensor([[0, 0, 31, 31], [0, 0, 63, 63]])
    print(cxcywh2xyxy(cxcywh))


def test_xyxy2cxcywh():
    cxcywh = t.Tensor([[0, 0, 15, 31], [150, 170, 200, 200]])
    print(xyxy2cxcywh(cxcywh))

import numpy as np
def get_wh_for_one_point(edge_len=32,
                         scales=np.array((2.0 ** 0, 2.0 ** (1.0 / 3.0), 2.0 ** (2.0 / 3.0))),
                         ratios=np.array((1.0 / 2.0, 1.0, 2.0))):
    anchors_num = len(scales) * len(ratios)
    anchors = np.zeros((anchors_num, 2))
    anchors[:, 0] = np.repeat(np.sqrt(ratios), len(scales))
    anchors[:, 1] = 1 / anchors[:, 0]
    anchors = edge_len * np.tile(scales, (1, len(ratios))).T * anchors
    return anchors


def get_cxcy_for_whole_fm(fm_shape, stride):
    x = (np.arange(0, fm_shape[0]) + 0.5) * stride[0]
    y = (np.arange(0, fm_shape[1]) + 0.5) * stride[1]
    mx, my = np.meshgrid(x, y)
    cxcy = np.stack((mx.ravel(), my.ravel()), axis=1)
    return cxcy


def get_anchors_for_whole_fm(
        origin_shape=(1024, 1024),
        fm_shape=(128, 128),
        edge_len=32,
        scales=np.array((2.0 ** 0, 2.0 ** (1.0 / 3.0), 2.0 ** (2.0 / 3.0))),
        ratios=np.array((1.0 / 2.0, 1.0, 2.0))):
    '''

    :param origin_shape:
    :param fm_shape:
    :param edge_len:
    :param scales:
    :param ratios:
    :return: order by cxcywh
    '''
    stride = (origin_shape[0] / fm_shape[0], origin_shape[1] / fm_shape[1])
    whs = get_wh_for_one_point(edge_len, scales, ratios)
    cxcys = get_cxcy_for_whole_fm(fm_shape, stride)
    anchors = np.zeros((len(cxcys) * len(whs), 4))
    anchors[:, 0:2] = cxcys.repeat(len(whs), axis=0)
    anchors[:, 2:4] = np.tile(whs, (len(cxcys), 1))
    return anchors
