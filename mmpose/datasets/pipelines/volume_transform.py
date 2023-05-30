# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
from mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform, get_warp_matrix,
                                         warp_affine_joints)
from mmpose.datasets.builder import PIPELINES
import torch.nn as nn
import torch.nn.functional as F
from mmpose.core.evaluation.top_down_eval import _get_max_preds_3d
from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe

@PIPELINES.register_module()
class Volume_rescale:
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    refer: https://github.com/runnanchen/SA-LSTM-3D-Landmark-Detection/blob/9ffedfb93c002f1a6f84a938f2afbecfe513f9b6/MyDataLoader.py
    """  
    def __init__(self, volume_size):
        self.volume_size = volume_size
        self.down = nn.Upsample(size=self.volume_size)

    def __call__(self, results):
        img = results['img']
        img = torch.unsqueeze(img,0)
        img = torch.unsqueeze(img,0)

        img_resize = self.down(img)
        img_resize = torch.squeeze(img_resize,0)
        img_resize = torch.squeeze(img_resize,0)

        results['img'] = img_resize
        return results
    
@PIPELINES.register_module()
class GenerateVoxel3DHeatmapTarget_Volume:
    """Generate the target 3d heatmap.

    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info_3d'.
    Modified keys: 'target', and 'target_weight'.

    Args:
        sigma: Sigma of heatmap gaussian (mm).
        joint_indices (list): Indices of joints used for heatmap generation.
            If None (default) is given, all joints will be used.
    """

    def __init__(self, sigma=50.0, joint_indices=None):
        self.sigma = sigma  # mm
        self.joint_indices = joint_indices

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        cfg = results['ann_info']
        #print('joint_3d:',joints_3d)
        #print('sigma:',self.sigma)
        num_joints = len(joints_3d)

        if self.joint_indices is not None:
            num_joints = len(self.joint_indices)
            joint_indices = self.joint_indices
        else:
            joint_indices = list(range(num_joints))

        width = results['width']
        height = results['height']
        channel = results['channel']
        space_size = [width,height,channel] ##mm
        space_center = [width/2,height/2,channel/2]
        cube_size = cfg['cube_size']
        grids_x = np.linspace(-space_size[0] / 2, space_size[0] / 2,
                              cube_size[0]) + space_center[0]
        grids_y = np.linspace(-space_size[1] / 2, space_size[1] / 2,
                              cube_size[1]) + space_center[1]
        grids_z = np.linspace(-space_size[2] / 2, space_size[2] / 2,
                              cube_size[2]) + space_center[2]
        target = np.zeros(
            (num_joints, cube_size[0], cube_size[1], cube_size[2]),
            dtype=np.float32)

        for idx, joint_id in enumerate(joint_indices):
            mu_x = joints_3d[joint_id][0]
            mu_y = joints_3d[joint_id][1]
            mu_z = joints_3d[joint_id][2]
            i_x = [
                np.searchsorted(grids_x, mu_x - (3 * self.sigma)),
                np.searchsorted(grids_x, mu_x + (3 * self.sigma), 'right')
            ]
            i_y = [
                np.searchsorted(grids_y, mu_y - 3 * self.sigma),
                np.searchsorted(grids_y, mu_y + 3 * self.sigma, 'right')
            ]
            i_z = [
                 np.searchsorted(grids_z, mu_z - 3 * self.sigma),
                np.searchsorted(grids_z, mu_z + 3 * self.sigma, 'right')
            ]
            if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                continue
            kernel_xs, kernel_ys, kernel_zs = np.meshgrid(
                grids_x[i_x[0]:i_x[1]],
                grids_y[i_y[0]:i_y[1]],
                grids_z[i_z[0]:i_z[1]],
                indexing='ij')
            g = np.exp(-((kernel_xs - mu_x)**2 + (kernel_ys - mu_y)**2 +
                        (kernel_zs - mu_z)**2) / (2 * self.sigma**2))
            target[idx, i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]] \
                = np.maximum(target[idx, i_x[0]:i_x[1],
                            i_y[0]:i_y[1], i_z[0]:i_z[1]], g)

        target = np.clip(target, 0, 1)
        if target.shape[0] == 1:
             target = target[0]

        # space_size = torch.tensor(space_size).clone().detach()
        # space_center = torch.tensor(space_center).clone().detach()
        # cube_size = torch.tensor(cube_size).clone().detach()
        # heat_point = []
        # for i, target_point in enumerate(target):
        #     target_point = torch.tensor(target_point)
        #     target_point = torch.unsqueeze(target_point,0)
        #     target_point = torch.unsqueeze(target_point,0).float()
        #     topk_values, topk_unravel_index = self._nms_by_max_pool(
        #         target_point.clone().detach())
        #     topk_unravel_index = self._get_real_locations(topk_unravel_index,space_size,cube_size,space_center)
        #     #print(i,': ',topk_unravel_index[0][0]) 
        #     heat_point.append(topk_unravel_index[0][0])
        # preds = np.stack(heat_point)
        # gts = np.stack(joints_3d)
        # print('heat_preds:',preds)
        # print('preds:',gts)
        # masks = np.ones_like(gts[:, 0], dtype=bool)
        # error = keypoint_mpjpe(preds, gts, masks,'none',1,1)
        # print('error:',error)
        # target = np.stack(target,axis=0)
        # target_cat = np.expand_dims(target,axis=0)
        # print('target_cat:',target_cat.shape)
        # preds, maxvals = _get_max_preds_3d(np.array(target_cat))
        # print(preds)
        # preds_ = []
        # for i in range(len(preds[0])):
        #     preds_temp = self._get_real_locations(torch.tensor(preds[0][i]),space_size,cube_size,space_center)
        #     preds_.append(preds_temp)
        # print('preds_:',preds_)
        # print('maxvals:',maxvals)
        #print(xxx)
        results['targets_3d'] = target
        
        return results

    @staticmethod
    def _get_3d_indices(indices, shape):
        """Get indices in the 3-D tensor.

        Args:
            indices (torch.Tensor(NXp)): Indices of points in the 1D tensor
            shape (torch.Size(3)): The shape of the original 3D tensor

        Returns:
            indices: Indices of points in the original 3D tensor
        """
        batch_size = indices.shape[0]
        num_people = indices.shape[1]
        indices_x = torch.div(indices,(shape[1] * shape[2]),rounding_mode='floor').reshape(batch_size, num_people, -1)
        indices_y = torch.div((indices % (shape[1] * shape[2])),shape[2],rounding_mode='floor').reshape(batch_size, num_people, -1)
        indices_z = (indices % shape[2]).reshape(batch_size, num_people, -1)
        indices = torch.cat([indices_x, indices_y, indices_z], dim=2)
        return indices

    def _get_real_locations(self, indices,space_size,cube_size,space_center):
        """
        Args:
            indices (torch.Tensor(NXP)): Indices of points in the 3D tensor

        Returns:
            real_locations (torch.Tensor(NXPx3)): Locations of points
                in the world coordinate system
        """
    
        real_locations = indices.float() / (
                cube_size - 1) * space_size + space_center - space_size / 2.0 
        return real_locations

    def _max_pool(self, inputs):
        kernel = 3
        padding = (kernel - 1) // 2
        max = F.max_pool3d(
            inputs, kernel_size=kernel, stride=1, padding=padding)
        keep = (inputs == max).float()
        return keep * inputs

    def _nms_by_max_pool(self, heatmap_volumes):
        max_num = 1
        batch_size = heatmap_volumes.shape[0]
        root_cubes_nms = self._max_pool(heatmap_volumes)
        root_cubes_nms_reshape = root_cubes_nms.reshape(batch_size, -1)
        topk_values, topk_index = root_cubes_nms_reshape.topk(max_num)
        #print('topk_index:',topk_index)
        topk_unravel_index = self._get_3d_indices(topk_index,
                                                  heatmap_volumes[0].shape)

        return topk_values, topk_unravel_index