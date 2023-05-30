# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import LOSSES
import torch.nn.functional as F
#from mmpose.models.heads.topdown_heatmap_base_head import TopdownHeatmapBaseHead


@LOSSES.register_module()
class JointsBCELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints * self.loss_weight

@LOSSES.register_module()
class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints * self.loss_weight

@LOSSES.register_module()
class JointsSmoothL1Loss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = F.smooth_l1_loss
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints * self.loss_weight

@LOSSES.register_module()
class JointsMSELoss_3D(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, target, target_weight):
        """Forward function."""

        loss = self.criterion(output, target)

        return loss
    
# @LOSSES.register_module()
# class JointsVarianceLoss(nn.Module):
#     """MSE loss for heatmaps.

#     Args:
#         use_target_weight (bool): Option to use weighted MSE loss.
#             Different joint types may have different target weights.
#         loss_weight (float): Weight of the loss. Default: 1.0.
#     """

#     def __init__(self, use_target_weight=False, loss_weight=1.):
#         super().__init__()

#         self.use_target_weight = use_target_weight
#         self.loss_weight = loss_weight

#     def forward(self, output, target, target_weight, img_size, img_metas):
#         """Forward function."""

#         pred_point = TopdownHeatmapBaseHead.decode(self, img_metas, output.numpy(),img_size=[img_width, img_height])['preds'] 
#         target_point = TopdownHeatmapBaseHead.decode(self, img_metas, target.numpy(),img_size=[img_width, img_height])['preds'] 
#         X_pred = pred_point[:,:,0]
#         Y_pred = pred_point[:,:,1]
#         X_target = target_point[:,:,0]
#         Y_target = target_point[:,:,1]
        
#         mean = _calc_distances(pred_point,target_point,target_weight.numpy().squeeze(-1) > 0)
        
        
#         #batch_size = output.size(0)

#         # num_joints = output.size(1)

#         # heatmaps_pred = output.reshape(
#         #     (batch_size, num_joints, -1)).split(1, 1)
#         # heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

#         # loss = 0.
#         # loss_std = []

#         # for idx in range(num_joints):
#         #     heatmap_pred = heatmaps_pred[idx].squeeze(1)
#         #     heatmap_gt = heatmaps_gt[idx].squeeze(1)
#         #     if self.use_target_weight:
#         #         loss += self.criterion(heatmap_pred * target_weight[:, idx],
#         #                                heatmap_gt * target_weight[:, idx])
#         #         loss_std.append(self.criterion(heatmap_pred * target_weight[:, idx],
#         #                                heatmap_gt * target_weight[:, idx]))
#         #     else:
#         #         loss += self.criterion(heatmap_pred, heatmap_gt)
#         #         loss_std.append(self.criterion(heatmap_pred, heatmap_gt))
#         # std_loss = torch.std(torch.tensor(loss_std)) * self.loss_weight
#         return std_loss


@LOSSES.register_module()
class CombinedTargetMSELoss(nn.Module):
    """MSE loss for combined target.
        CombinedTarget: The combination of classification target
        (response map) and regression target (offset map).
        Paper ref: Huang et al. The Devil is in the Details: Delving into
        Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_channels = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        loss = 0.
        num_joints = num_channels // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx * 3].squeeze()
            heatmap_gt = heatmaps_gt[idx * 3].squeeze()
            offset_x_pred = heatmaps_pred[idx * 3 + 1].squeeze()
            offset_x_gt = heatmaps_gt[idx * 3 + 1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()
            if self.use_target_weight:
                heatmap_pred = heatmap_pred * target_weight[:, idx]
                heatmap_gt = heatmap_gt * target_weight[:, idx]
            # classification loss
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            # regression loss
            loss += 0.5 * self.criterion(heatmap_gt * offset_x_pred,
                                         heatmap_gt * offset_x_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_y_pred,
                                         heatmap_gt * offset_y_gt)
        return loss / num_joints * self.loss_weight


@LOSSES.register_module()
class JointsOHKMMSELoss(nn.Module):
    """MSE loss with online hard keypoint mining.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        topk (int): Only top k joint losses are kept.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, topk=8, loss_weight=1.):
        super().__init__()
        assert topk > 0
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.loss_weight = loss_weight

    def _ohkm(self, loss):
        """Online hard keypoint mining."""
        ohkm_loss = 0.
        N = len(loss)
        for i in range(N):
            sub_loss = loss[i]
            _, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= N
        return ohkm_loss

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)
        if num_joints < self.topk:
            raise ValueError(f'topk ({self.topk}) should not '
                             f'larger than num_joints ({num_joints}).')
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        losses = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                losses.append(
                    self.criterion(heatmap_pred * target_weight[:, idx],
                                   heatmap_gt * target_weight[:, idx]))
            else:
                losses.append(self.criterion(heatmap_pred, heatmap_gt))

        losses = [loss.mean(dim=1).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)

        return self._ohkm(losses) * self.loss_weight