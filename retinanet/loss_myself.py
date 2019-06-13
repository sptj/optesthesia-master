import torch as t
import torch.nn as nn
import torch.nn.functional as F


class focal_loss(nn.Module):
    '''
    :param x: x is the probablity indicate target.
    :param y: y is the ground truth of target
    :return: loss value
    '''

    def __init__(self, num_cls):
        super(focal_loss, self).__init__()
        self.alpha = 0.25
        self.gama = 2
        self.num_cls = num_cls

    def focal_loss(self, x, y):
        x = x.sigmoid()
        y = y.float()
        pt = y * x + (1 - y) * (1 - x)
        alpha_spec = y * self.alpha + (1 - y) * (1 - self.alpha)
        FL = -alpha_spec * t.pow((1 - pt), self.gama) * t.log(pt)
        return FL.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        target_mask = cls_targets > 0
        masked_for_loc = target_mask.unsqueeze(2).expand_as(loc_preds)
        exec_loc_pred = loc_preds[masked_for_loc].view(-1, 4)
        exec_loc_targ = loc_targets[masked_for_loc].view(-1, 4).float().cuda()
        loc_loss = F.smooth_l1_loss(exec_loc_pred, exec_loc_targ, size_average=False)
        '''
        caution:
        cls_target 里边存储的是数字
        cls_pred   里边存储的是独热码
        '''
        unignored_mask = cls_targets > -1
        mask_for_cls = unignored_mask.unsqueeze(2).expand_as(cls_preds)
        masked_cls_pred = cls_preds[mask_for_cls].view(-1, self.num_cls)
        # ==============================================================
        masked_cls_target = cls_targets[cls_targets > -1].view(-1, self.num_cls)

        masked_cls_target = t.eye(1 + self.num_cls)[masked_cls_target][:, :, 1:].view(-1, self.num_cls)

        cls_loss = self.focal_loss(masked_cls_pred, masked_cls_target.cuda())
        # ==============================================================
        total_loss = (loc_loss + cls_loss) / target_mask.sum()

        # print("loc_loss         ",loc_loss)
        # print("cls_loss         ",cls_loss)
        # print("target_mask.sum()",target_mask.sum())
        return total_loss


if __name__ == '__main__':
    pass
