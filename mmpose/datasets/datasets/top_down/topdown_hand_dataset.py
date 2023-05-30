# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from collections import OrderedDict, defaultdict

import json_tricks as json
import numpy as np
from mmcv import Config
from xtcocotools.cocoeval import COCOeval

from ....core.post_processing import oks_nms, soft_oks_nms
from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset
from mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform, get_warp_matrix,
                                         warp_affine_joints)

from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe,
                                                  keypoint_nme, keypoint_epe_std,
                                                  keypoint_pck_accuracy,keypoint_epe_point,
                                                  )


@DATASETS.register_module()
class TopDownHandDataset(Kpt2dSviewRgbImgTopDownDataset):
    """ISBI in COCO format with 19 keypoints

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
            cfg = Config.fromfile('configs/_base_/datasets/coco.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']
        self.image_size = data_cfg['image_size']

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        if (not self.test_mode) or self.use_gt_bbox:
            gt_db = self._load_coco_keypoint_annotations()
        else:
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]

        Args:
            img_id: coco image id

        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        pixel_spacing = img_ann['pixel_spacing']

        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)
        valid_objs = []
        for obj in objs:
            x, y, w, h = 0,0,width,height
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
            objs = valid_objs
        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])
            center, scale = self._xywh2cs(0, 0, width, height, padding=1.25)
            image_file = os.path.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id,
                'img_size_ori':[width,height],
                'pixel_spacing':pixel_spacing,
            })
            bbox_id = bbox_id + 1
        return rec

    def _load_coco_person_detection_results(self):
        """Load coco person detection results."""
        num_joints = self.ann_info['num_joints']
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            raise ValueError('=> Load %s fail!' % self.bbox_file)

        print(f'=> Total boxes: {len(all_boxes)}')

        kpt_db = []
        bbox_id = 0
        for det_res in all_boxes:
            if det_res['category_id'] != 1:
                continue

            image_file = os.path.join(self.img_prefix,
                                      self.id2name[det_res['image_id']])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.det_bbox_thr:
                continue

            center, scale = self._xywh2cs(*box[:4])
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)
            kpt_db.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'rotation': 0,
                'bbox': box[:4],
                'bbox_score': score,
                'dataset': self.dataset_name,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1
        print(f'=> Total boxes after filter '
              f'low score@{self.det_bbox_thr}: {bbox_id}')
        return kpt_db

    def evaluate(self, outputs, res_folder, metric='MRE', **kwargs):
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
        if len(metrics[0].split(',')) > 1:
            metrics = metrics[0].split(',')
        allowed_metrics = ['MRE_i2','MRE_std_i2','SDR_2_i2','SDR_2.5_i2','SDR_3_i2','SDR_4_i2','SDR_10_i2']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        kpts = defaultdict(list)
        for output in outputs:
            preds = output['preds']
            boxes = output['boxes']
            image_paths = output['image_paths']
            bbox_ids = output['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                img_name = image_paths[i][len(self.img_prefix):].split('/')[-1]
                image_id = self.name2id[img_name]
                kpts[image_id].append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i],
                })
        kpts = self._sort_and_unique_bboxes(kpts)
        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        return name_value

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts

    def _report_metric(self,
                       res_file,
                       metrics,
                       image_size=None,
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
            center = item['center']
            scale = item['scale']
            rotation = item['rotation']
            pixel_spacing = float(item['pixel_spacing'])
            if image_size is not None:
                trans = get_affine_transform(center,scale,rotation,image_size)
                predict = np.array(pred['keypoints'])[:, :-1]
                gt = np.array(item['joints_3d'])[:, :-1]
                for i in range(len(predict)):
                    predict[i] = affine_transform(predict[i, 0:2], trans)
                    gt[i] = affine_transform(gt[i, 0:2], trans)
                outputs.append(predict*pixel_spacing)
                gts.append(gt*pixel_spacing)
            else:
                outputs.append(np.array(pred['keypoints'])[:, :-1]*pixel_spacing)
                gts.append(np.array(item['joints_3d'])[:, :-1]*pixel_spacing)
            masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            box_sizes.append(item.get('box_size', 1))
        outputs = np.array(outputs)
        N = outputs.shape[0]
        gts = np.array(gts)     #pixel_spacing
        masks = np.array(masks)
        threshold_bbox = np.array(threshold_bbox)
        threshold_head_box = np.array(threshold_head_box)
        box_sizes = np.array(box_sizes).reshape([-1, 1])
        if 'SDR' in metrics:
            _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                              threshold_bbox)
            info_str.append(('SDR', pck))
        if 'SDR_2_i2' in metrics:
             _, pck_2_i2, _ = keypoint_pck_accuracy(outputs, gts, masks,thr=2,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('SDR_2_i2', pck_2_i2))
        if 'SDR_2.5_i2' in metrics:
             _, pck_25_i2, _ = keypoint_pck_accuracy(outputs, gts, masks,thr=2.5,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('SDR_2.5_i2', pck_25_i2))
        if 'SDR_3_i2' in metrics:
             _, pck_3_i2, _ = keypoint_pck_accuracy(outputs, gts, masks,thr=3,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('SDR_3_i2', pck_3_i2))
        if 'SDR_4_i2' in metrics:
             _, pck_4_i2, _ = keypoint_pck_accuracy(outputs, gts, masks,thr=4,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('SDR_4_i2', pck_4_i2))
        if 'SDR_10_i2' in metrics:
             _, pck_10_i2, _ = keypoint_pck_accuracy(outputs, gts, masks,thr=10,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('SDR_10_i2', pck_10_i2))
        if 'MRE_i2' in metrics:
            info_str.append(('MRE_i2', keypoint_epe(outputs, gts, masks)))
        if 'MRE' in metrics:
            info_str.append(('MRE', keypoint_epe(outputs/pixel_spacing, gts/pixel_spacing, masks)))
        if 'MRE_std_i2' in metrics:
            info_str.append(('MRE_std_i2', keypoint_epe_std(outputs, gts, masks)))
            
        if 'MRE_wise_point_i2' in metrics:
            info_str.append(('MRE_wise_point_i2', keypoint_epe_point(outputs, gts, masks,self.keypoint_info)))

        if 'NME' in metrics:
            normalize_factor = self._get_normalize_factor(
                gts=gts, box_sizes=box_sizes)
            info_str.append(
                ('NME', keypoint_nme(outputs, gts, masks, normalize_factor)))
        return info_str
