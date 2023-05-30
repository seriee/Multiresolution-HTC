# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod

import json_tricks as json
import numpy as np
from torch.utils.data import Dataset
from xtcocotools.coco import COCO

from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe,
                                                  keypoint_nme, keypoint_epe_std,
                                                  keypoint_pck_accuracy,keypoint_epe_point,
                                                  )
from mmpose.core.post_processing import (affine_transform, get_affine_transform)
from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose


class Kpt2dSviewRgbImgTopDownDataset(Dataset, metaclass=ABCMeta):
    """Base class for keypoint 2D top-down pose estimation with single-view RGB
    image as the input.

    All fashion datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_db`, 'evaluate'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        coco_style (bool): Whether the annotation json is coco-style.
            Default: True
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 coco_style=True,
                 test_mode=False):

        self.image_info = {}
        self.ann_info = {}

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['num_output_channels'] = data_cfg['num_output_channels']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        self.ann_info['use_different_joint_weights'] = data_cfg.get(
            'use_different_joint_weights', False)

        if dataset_info is None:
            raise ValueError(
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.')
#        print('dataset_info: ',dataset_info)
        dataset_info = DatasetInfo(dataset_info)
        self.keypoint_info = dataset_info.keypoint_info
        assert self.ann_info['num_joints'] == dataset_info.keypoint_num
        self.ann_info['flip_pairs'] = dataset_info.flip_pairs
        self.ann_info['flip_index'] = dataset_info.flip_index
        self.ann_info['upper_body_ids'] = dataset_info.upper_body_ids
        self.ann_info['lower_body_ids'] = dataset_info.lower_body_ids
        self.ann_info['joint_weights'] = dataset_info.joint_weights
        if 'skeleton' in self.ann_info:
            self.ann_info['skeleton'] = dataset_info.skeleton
        if 'space_size' in self.ann_info:
            self.ann_info['space_size'] = dataset_info.space_size
        self.sigmas = dataset_info.sigmas
        self.dataset_name = dataset_info.dataset_name

        if coco_style:
            self.coco = COCO(ann_file)
            if 'categories' in self.coco.dataset:
                cats = [
                    cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())
                ]
                self.classes = ['__background__'] + cats
                self.num_classes = len(self.classes)
                self._class_to_ind = dict(
                    zip(self.classes, range(self.num_classes)))
                self._class_to_coco_ind = dict(
                    zip(cats, self.coco.getCatIds()))
                self._coco_ind_to_class_ind = dict(
                    (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                    for cls in self.classes[1:])
            self.img_ids = self.coco.getImgIds()
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id = self._get_mapping_id_name(
                self.coco.imgs)

        self.db = []

        self.pipeline = Compose(self.pipeline)

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _xywh2cs(self, x, y, w, h, padding=1.25):
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h (float): left, top, width and height
            padding (float): bounding box padding factor

        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
            'image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if (not self.test_mode) and np.random.rand() < 0.3:
            center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]
    
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * padding
        return center, scale

    def _get_normalize_factor(self, gts, *args, **kwargs):
        """Get the normalize factor. generally inter-ocular distance measured
        as the Euclidean distance between the outer corners of the eyes is
        used. This function should be overrode, to measure NME.

        Args:
            gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

        Returns:
            np.ndarray[N, 2]: normalized factor
        """
        return np.ones([gts.shape[0], 2], dtype=np.float32)

    @abstractmethod
    def _get_db(self):
        """Load dataset."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, cfg, outputs, res_folder, metric, *args, **kwargs):
        """Evaluate keypoint results."""
        raise NotImplementedError

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

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
            if image_size is not None:
                trans = get_affine_transform(center,scale,rotation,image_size)
                predict = np.array(pred['keypoints'])[:, :-1]
                gt = np.array(item['joints_3d'])[:, :-1]
                for i in range(len(predict)):
                    predict[i] = affine_transform(predict[i, 0:2], trans)
                    gt[i] = affine_transform(gt[i, 0:2], trans)
                outputs.append(predict)
                gts.append(gt)
            else:
                outputs.append(np.array(pred['keypoints'])[:, :-1])
                gts.append(np.array(item['joints_3d'])[:, :-1])
                masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)
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

        if 'SDR_2h' in metrics:
             _, pck_2, _ = keypoint_pck_accuracy(outputs*0.66, gts*0.66, masks,thr=2,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('SDR_2h', pck_2))
        if 'SDR_2.5h' in metrics:
             _, pck_25, _ = keypoint_pck_accuracy(outputs*0.66, gts*0.66, masks,thr=2.5,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('SDR_2.5h', pck_25))
        if 'SDR_3h' in metrics:
             _, pck_3, _ = keypoint_pck_accuracy(outputs*0.66, gts*0.66, masks,thr=3,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('SDR_3h', pck_3))
        if 'SDR_4h' in metrics:
             _, pck_4, _ = keypoint_pck_accuracy(outputs*0.66, gts*0.66, masks,thr=4,normalize=np.ones((N, 2), dtype=np.float32))
             info_str.append(('SDR_4h', pck_4))
        if 'MRE_h' in metrics:
            info_str.append(('MRE_h', keypoint_epe(outputs*0.66, gts*0.66, masks)))
        if 'MRE_std_h' in metrics:
            info_str.append(('MRE_std_h', keypoint_epe_std(outputs*0.66, gts*0.66, masks)))
       
        return info_str

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.db)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = copy.deepcopy(self.db[idx])
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts
