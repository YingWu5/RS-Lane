"""
get_tp_fp_fn, SoftDiceLoss, and DC_and_CE/TopK_loss are from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch import einsum
import numpy as np

def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
    
        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = 1 / (einsum("bcxyz->bc", y_onehot).type(torch.float32) + 1e-10)**2
        intersection: torch.Tensor = w * einsum("bcxyz, bcxyz->bc", net_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxyz->bc", net_output) + einsum("bcxyz->bc", y_onehot))
        divided: torch.Tensor =  - 2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc



def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)

class GDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLossV2, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        input = flatten(net_output)
        target = flatten(y_onehot)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.smooth), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return  - 2. * intersect / denominator.clamp(min=self.smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1606.04797.pdf
        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc 


class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = 0.3
        self.beta = 0.7

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)


        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return -tversky

class FocalTversky_loss(nn.Module):
    """
    paper: https://arxiv.org/pdf/1810.07842.pdf
    author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py#L65
    """
    def __init__(self, tversky_kwargs, gamma=0.75):
        super(FocalTversky_loss, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(**tversky_kwargs)

    def forward(self, net_output, target):
        tversky_loss = 1 + self.tversky(net_output, target) # = 1-tversky(net_output, target)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky


class PenaltyGDiceLoss(nn.Module):
    """
    paper: https://openreview.net/forum?id=H1lTh8unKN
    """
    def __init__(self, gdice_kwargs):
        super(PenaltyGDiceLoss, self).__init__()
        self.k = 2.5
        self.gdc = GDiceLoss(apply_nonlin=softmax_helper, **gdice_kwargs)

    def forward(self, net_output, target):
        gdc_loss = self.gdc(net_output, target)
        penalty_gdc = gdc_loss / (1 + self.k * (1 - gdc_loss))

        return penalty_gdc




