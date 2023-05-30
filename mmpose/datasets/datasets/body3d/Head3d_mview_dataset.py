# Copyright (c) OpenMMLab. All rights reserved.
import copy
import glob
import json
import os
import os.path as osp
import pickle
from tkinter import image_names
import warnings
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
from mmcv import Config

from mmpose.core.camera import SimpleCamera
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt3dMviewRgbImgDirectDataset
from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe,
                                                  keypoint_nme, keypoint_epe_std,
                                                  keypoint_pck_accuracy,keypoint_epe_point,
                                                  )


@DATASETS.register_module()
class Head3DMviewDataset(Kpt3dMviewRgbImgDirectDataset):
    """
    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """
    ALLOWED_METRICS = {'mpjpe', 'mAP','EPE_std','EPE'}

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/panoptic_body3d.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.load_config(data_cfg)
        self.ann_info['use_different_joint_weights'] = False
        self.ann_file = ann_file
        self.img_prefix = img_prefix

        # if ann_file is None:
        #     self.db_file = os.path.join(
        #         img_prefix, f'group_{self.subset}_cam{self.num_cameras}.pkl')
        # else:
        #     self.db_file = ann_file

        # if osp.exists(self.db_file):
        #     with open(self.db_file, 'rb') as f:
        #         info = pickle.load(f)
        #     assert info['cam_list'] == self.cam_list
        #     self.db = info['db']
        #else:
        self.db = self._get_db()
        info = {
            'cam_list': self.cam_list,
            'db': self.db
        }
        # with open(self.db_file, 'wb') as f:
        #     pickle.dump(info, f)

        self.db_size = len(self.db)

        print(f'=> load {len(self.db)} samples')

    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.num_joints = data_cfg['num_joints']
        assert self.num_joints <= 16
        self.cam_list = data_cfg['cam_list']
        self.num_cameras = data_cfg['num_cameras']
        assert self.num_cameras == len(self.cam_list)
        self.subset = data_cfg.get('subset', 'train')
        self.need_camera_param = True
        self.max_persons = data_cfg.get('max_num', 1)

    def _get_scale(self, raw_image_size):
        heatmap_size = self.ann_info['heatmap_size']
        image_size = self.ann_info['image_size']
        assert heatmap_size[0] / heatmap_size[1] \
               == image_size[0] / image_size[1]
        w, h = raw_image_size
        w_resized, h_resized = image_size
        if w / w_resized < h / h_resized:
            w_pad = h / h_resized * w_resized
            h_pad = h
        else:
            w_pad = w
            h_pad = w / w_resized * h_resized

        scale = np.array([w_pad, h_pad], dtype=np.float32)

        return scale

    def _get_cam(self, cam_param, num):
        """Get camera parameters.

        Args:
            seq (str): Sequence name.

        Returns: Camera parameters.
        """
        calib = cam_param
        
        cam_list = list(range(num*self.num_cameras,num*self.num_cameras+self.num_cameras))
        M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib:
            if (cam['image_id']) in (cam_list):
                #print(cam['image_id'])
                sel_cam = {}
                R_w2c = np.array(cam['R']).dot(M)
                T_w2c = np.array(cam['t']).reshape((3, 1))  
                R_c2w = R_w2c.T
                T_c2w = -R_w2c.T @ T_w2c
                sel_cam['R'] = R_c2w.tolist()
                sel_cam['T'] = T_c2w.tolist()
                sel_cam['K'] = cam['K'][:2]
                distCoef = cam['distCoef']
                sel_cam['k'] = [distCoef[0], distCoef[1], distCoef[4]]
                sel_cam['p'] = [distCoef[2], distCoef[3]]
                cameras[cam['image_id']] = sel_cam

        return cameras

    def _get_db(self):
        """Get dataset base.

        Returns:
            dict: the dataset base (2D and 3D information)
        """
        db = []
        sample_id = 0
        #print(self.ann_file)
        with open(self.ann_file) as dfile:
            ann_file = json.load(dfile)
        camera_param = ann_file['camera']
        annotation = ann_file['annotations']
        self.image = ann_file['images']
        num_sequence = int(len(annotation)/self.num_cameras)
        # print('annotation:',len(annotation))
        #print('num_sequence:',num_sequence)
        for i in range(num_sequence):
            cam_params = self._get_cam(camera_param,i)
            for k, cam_param in cam_params.items():
                single_view_camera = SimpleCamera(cam_param)
                img_file = self.img_prefix+self.image[k]['file_name']
                width = self.image[k]['width']
                height = self.image[k]['height']
                pose_3d = np.array([annotation[k]['keypoint_3d']]).astype('float32')
                all_poses_vis_3d = np.ones((self.max_persons, self.num_joints, 3),dtype=np.float32)
                pose_2d = np.array([annotation[k]['keypoints']]).astype('float32')
                all_roots_3d = np.zeros((self.max_persons, 3),dtype=np.float32)
                #print(pose_2d.shape)
                pose_2d[:,:,-1] = 2
                keypoints_2d = np.zeros((self.num_joints, 3), dtype=np.float32)
                keypoints_2d_visible = np.zeros((self.num_joints, 3), dtype=np.float32)
                keypoints = pose_2d[0]
                keypoints_2d[:, :2] = keypoints[:, :2]
                keypoints_2d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])
                c_2d, s_2d = self._xywh2cs(0, 0, width, height)
                #motion_mat = annotation[k]['motion_mat']
                #project_mat = annotation[k]['proj_mat']
                db.append({
                    'image_file': img_file,
                    'joints_3d': pose_3d,
                    'person_ids':0,
                    'joints_3d_visible':all_poses_vis_3d,
                    'joints_2d_visible':keypoints_2d_visible,
                    'joints_2d':keypoints_2d,
                    'roots_3d':all_roots_3d,
                    'camera': cam_param,
                    'num_persons': 1,
                    'sample_id':0,
                    'center':np.array((width / 2, height / 2),
                                         dtype=np.float32),
                    'scale':self._get_scale((width, height)),
                    'rotation':0,
                    'c_2d':c_2d,
                    's_2d':s_2d,
                    'bbox_id': 0,
                    #'motion_mat':motion_mat,
                    #'projection_mat':project_mat
                })         
        return db

    def evaluate(self, outputs, res_folder, metric='EPE', **kwargs):
        """Evaluate coco keypoint results. The pose prediction results will be
        saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            outputs (list[dict]): Outputs containing the following items.

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['data/coco/val2017\
                    /000000393226.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap
                - bbox_id (list(int)).
            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
#        print('metrics:', metrics)
        if len(metrics[0].split(',')) > 1:
            metrics = metrics[0].split(',')
#        print('new_metrics:',met)
        allowed_metrics = ['EPE_std','EPE','PCK_2','PCK_2.5','PCK_3','PCK_4']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')
        #print(len(self.image))
        kpts = defaultdict(list)
        #print('output:',len(outputs))
        for output in outputs:
            nview = len(output)
            #print('nview:',nview)
            for x in range(nview):
                #print('output:',output)
                preds = output[x]['preds']
                boxes = output[x]['boxes']
                #print('preds:',preds.shape)
                #print('boxes:',boxes.shape)
                image_paths = output[x]['image_paths']
                batch_size = len(image_paths)
                bbox_ids = output[x]['bbox_ids']
                #print('bbox_ids:',bbox_ids)
                #print('batch_size:',batch_size)
                for i in range(batch_size):
                    #print('i:',i)
                    #print(image_paths[i])
                    image_id = self.name2id[image_paths[i][len(self.img_prefix):]]
                    img_name = image_paths[i][0].split('/')[-1]
                    #print(img_name.split('/')[-1])
                    # for z in range(len(self.image)):
                    #     #print(self.image[z]['file_name'])
                    #     if self.image[z]['file_name'] == img_name:
                    #         image_id = z
                    kpts[image_id].append({
                        'keypoints': preds[i].tolist(),
                        'center': boxes[i][0:2].tolist(),
                        'scale': boxes[i][2:4].tolist(),
                        'area': float(boxes[i][4]),
                        'score': float(boxes[i][5]),
                        'image_id': image_id,
                        'bbox_id': bbox_ids[i],
                        'img_name':img_name
                    })
        #print('kpts:',kpts)
        kpts = self._sort_and_unique_bboxes(kpts)
#        print('kpts:',len(kpts))
        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        return name_value

    def evaluate_3d(self, outputs, res_folder, metric='mpjpe', **kwargs):
        """

        Args:
            outputs list(dict(pose_3d, sample_id)):
                pose_3d (np.ndarray): predicted 3D human pose
                sample_id (np.ndarray): sample id of a frame.
            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Defaults: 'mpjpe'.
            **kwargs:

        Returns:

        """
        _outputs = np.concatenate([output['pose_3d'] for output in outputs],
                                   axis=0)

        # print('pose_3ds:',pose_3ds.shape)
        # sample_ids = []
        # for output in outputs:
        #     sample_ids.extend(output['sample_id'])
        # for (sample_id, pose_3d) in zip(sample_ids, pose_3ds):
        #     print('sample_id:',sample_id)
        #     print('pose_3ds:',pose_3ds.shape)
        # _outputs = [
        #     dict(sample_id=sample_id, pose_3d=pose_3d)
        #     for (sample_id, pose_3d) in zip(sample_ids, pose_3ds)
        # ]
        # _outputs = self._sort_and_unique_outputs(_outputs, key='sample_id')
  
        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}"'
                    f'Supported metrics are {self.ALLOWED_METRICS}')

        res_file = osp.join(res_folder, 'result_keypoints.json')
   
        mmcv.dump(_outputs, res_file)

        eval_list = []
        gt_num = self.db_size // self.num_cameras
        assert len(
            _outputs) == gt_num, f'number mismatch: {(_outputs.shape)[0]}, {gt_num}'

        total_gt = 0
        for i in range(gt_num):
            index = self.num_cameras * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_visible']

            if joints_3d_vis.sum() < 1:
                continue

            pred = _outputs[i,:,:,:].copy()
            #print(pred)
            #pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    if vis.sum() < 1:
                        break
                    mpjpe = np.mean(
                        np.sqrt(
                            np.sum((pose[vis, 0:3] - gt[vis])**2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    'mpjpe': float(min_mpjpe),
                    'score': float(score),
                    'gt_id': int(total_gt + min_gt)
                })

            total_gt += (joints_3d_vis[:, :, 0].sum(-1) >= 1).sum()

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        ars = []
        for t in mpjpe_threshold:
            ap, ar = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            ars.append(ar)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                stats_names = ['RECALL 500mm', 'MPJPE']
                info_str = list(
                    zip(stats_names, [
                        self._eval_list_to_recall(eval_list, total_gt),
                        self._eval_list_to_mpjpe(eval_list)
                    ]))
            elif _metric == 'mAP':
                stats_names = [
                    'AP 25', 'AP 50', 'AP 75', 'AP 100', 'AP 125', 'AP 150',
                    'mAP', 'AR 25', 'AR 50', 'AR 75', 'AR 100', 'AR 125',
                    'AR 150', 'mAR'
                ]
                mAP = np.array(aps).mean()
                mAR = np.array(ars).mean()
                info_str = list(zip(stats_names, aps + [mAP] + ars + [mAR]))
            else:
                raise NotImplementedError
            name_value_tuples.extend(info_str)

        return OrderedDict(name_value_tuples)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        """Get Average Precision (AP) and Average Recall at a certain
        threshold."""

        eval_list.sort(key=lambda k: k['score'], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
                tp[i] = 1
                gt_det.append(item['gt_id'])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        """Get MPJPE within a certain threshold."""
        #eval_list.sort(key=lambda k: k['score'], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            #if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
            mpjpes.append(item['mpjpe'])
            gt_det.append(item['gt_id'])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        """Get Recall at a certain threshold."""
        gt_ids = [e['gt_id'] for e in eval_list if e['mpjpe'] < threshold]

        return len(np.unique(gt_ids)) / total_gt

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = {}
        print('self.db:',len(self.db))
        for c in range(self.num_cameras):
            result = copy.deepcopy(self.db[self.num_cameras * idx + c])
            result['ann_info'] = self.ann_info
            width = 1920
            height = 1080
            result['mask'] = [np.ones((height, width), dtype=np.float32)]
            results[c] = result

        return self.pipeline(results)

    @staticmethod
    def _sort_and_unique_outputs(outputs, key='sample_id'):
        """sort outputs and remove the repeated ones."""
        outputs = sorted(outputs, key=lambda x: x[key])
        num_outputs = len(outputs)
        for i in range(num_outputs - 1, 0, -1):
            if outputs[i][key] == outputs[i - 1][key]:
                del outputs[i]

        return outputs

    def _report_metric(self,
                       res_file,
                       metrics,
                       pck_thr=0.2,
                       pckh_thr=0.7,
                       auc_nor=30):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE', 'NME'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        outputs = []
        gts = []
        masks = []
        box_sizes = []
        threshold_bbox = []
        threshold_head_box = []
        for pred, item in zip(preds.values(), self.db):
            pred = pred[0]
            #print('pred:',pred)
            #print('gt:',np.array(item['joints_3d'])[:, :-1])
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_2d'])[:, :-1])
            masks.append((np.array(item['joints_2d_visible'])[:, 0]) > 0)
#            if 'PCK' in metrics:
#                bbox = np.array(item['bbox'])
#                bbox_thr = np.max(bbox[2:])
#                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            if 'PCKh' in metrics:
                head_box_thr = item['head_size']
                threshold_head_box.append(
                    np.array([head_box_thr, head_box_thr]))
            box_sizes.append(item.get('box_size', 1))

        outputs = np.array(outputs)
        N = outputs.shape[0]
        gts = np.array(gts)
        masks = np.array(masks)
        threshold_bbox = np.array(threshold_bbox)
        threshold_head_box = np.array(threshold_head_box)
        box_sizes = np.array(box_sizes).reshape([-1, 1])
#        print('\n metrics:',metrics)
        if 'PCK' in metrics:
            _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                              threshold_bbox)
            info_str.append(('PCK', pck))
        if 'PCK_2' in metrics:
             _, pck_2, _ = keypoint_pck_accuracy(outputs, gts, masks,thr=2,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('PCK_2', pck_2))
        if 'PCK_2.5' in metrics:
             _, pck_25, _ = keypoint_pck_accuracy(outputs, gts, masks,thr=2.5,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('PCK_2.5', pck_25))
        if 'PCK_3' in metrics:
             _, pck_3, _ = keypoint_pck_accuracy(outputs, gts, masks,thr=3,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('PCK_3', pck_3))
        if 'PCK_4' in metrics:
             _, pck_4, _ = keypoint_pck_accuracy(outputs, gts, masks,thr=4,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('PCK_4', pck_4))
        if 'PCKh' in metrics:
            _, pckh, _ = keypoint_pck_accuracy(outputs, gts, masks, pckh_thr,
                                               threshold_head_box)
            info_str.append(('PCKh', pckh))

        if 'AUC' in metrics:
            info_str.append(('AUC', keypoint_auc(outputs, gts, masks,
                                                 auc_nor)))
        if 'EPE' in metrics:
            info_str.append(('EPE', keypoint_epe(outputs, gts, masks)))

        if 'EPE_std' in metrics:
            info_str.append(('EPE_std', keypoint_epe_std(outputs, gts, masks)))
            
        if 'EPE_wise_point' in metrics:
            info_str.append(('EPE_wise_point', keypoint_epe_point(outputs, gts, masks,self.keypoint_info)))

        if 'NME' in metrics:
            normalize_factor = self._get_normalize_factor(
                gts=gts, box_sizes=box_sizes)
            info_str.append(
                ('NME', keypoint_nme(outputs, gts, masks, normalize_factor)))
#        print('info_str:',info_str)
        return info_str

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts