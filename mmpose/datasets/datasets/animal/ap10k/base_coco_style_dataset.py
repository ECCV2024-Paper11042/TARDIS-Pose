# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import logging
from copy import deepcopy
from itertools import filterfalse, groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt

import numpy as np
import cv2
import json
from tqdm import tqdm

from mmengine.dataset import BaseDataset, force_full_init
from mmengine.fileio import exists, get_local_path, load
from mmengine.utils import is_list_of
from mmengine.logging import print_log
from xtcocotools.coco import COCO

from mmpose.structures.bbox import bbox_xywh2xyxy
from mmpose.datasets.datasets.utils import parse_pose_metainfo

import albumentations as alb
# from albumentations.pytorch import transforms as alb_torch

def load_dino_hdf5(filepath: str) -> h5py.File:
    print(f"Loading DINO features from file {filepath}")
    return h5py.File(filepath, 'r')


class BaseCocoStyleDataset(BaseDataset):
    """Base class for COCO-style datasets.

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data.
            Default: ``dict(img='')``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict()

    def __init__(self,
                 ann_file: str = '',
                 bbox_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 transform=None,
                 image_list_file=None,
                 # load_unlabeled=False,
                 image_annotation_type='labeled',
                 dino_file: Optional[str] = None
                 ):

        if data_mode not in {'topdown', 'bottomup'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid data_mode: '
                f'{data_mode}. Should be "topdown" or "bottomup".')
        self.data_mode = data_mode

        if bbox_file:
            if self.data_mode != 'topdown':
                raise ValueError(
                    f'{self.__class__.__name__} is set to {self.data_mode}: '
                    'mode, while "bbox_file" is only '
                    'supported in topdown mode.')

            if not test_mode:
                raise ValueError(
                    f'{self.__class__.__name__} has `test_mode==False` '
                    'while "bbox_file" is only '
                    'supported when `test_mode==True`.')
        self.bbox_file = bbox_file

        if transform is None:
            transform = alb.Compose([])
        self.transform = transform

        self.img_ids = None
        self.labeled_ids = None
        # self.load_unlabeled = load_unlabeled
        self.image_annotation_type = image_annotation_type

        if image_list_file is not None:
            with open(image_list_file, 'r') as fp:
                self.labeled_ids = json.load(fp)

        self.dino_file = dino_file
        # if not hasattr(self, 'attentions'):
        self.attentions = None
        if self.dino_file is not None:
            assert osp.isfile(self.dino_file), f"Could not find DINO feature file {self.dino_file}"
            self.attentions = load_dino_hdf5(self.dino_file)

        # from pose.extract_dino import PCAVis
        # self.dinovis = PCAVis(segment=False)
        # self.dinovis.load("../data/dino/dino_train-split1_vits14_14.h5")

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)


    def _load_data_into_memory(self) -> None:
        print(f'Pre-fetching all images...')
        self.data_cache = {}
        for data_info in tqdm(self.data_list):
            filename = data_info['img_path']
            img = cv2.imread(filename)
            data_info['img'] = img


    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
              ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        if len(self.data_list) < 100:
            self._load_data_into_memory()

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True


    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """

        if metainfo is None:
            metainfo = deepcopy(cls.METAINFO)

        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        # parse pose metainfo if it has been assigned
        if metainfo:
            metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    @force_full_init
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        :class:`BaseCocoStyleDataset` overrides this method from
        :class:`mmengine.dataset.BaseDataset` to add the metainfo into
        the ``data_info`` before it is passed to the pipeline.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        sample = self._get_sample(idx)

        # sample['keypoints_'] = sample['keypoints']
        # sample['keypoints'] = sample.pop('transformed_keypoints')[0,:, :2]
        # if 'transformed_keypoints_visible' in sample:
        #     sample['keypoints_visible'] = sample['transformed_keypoints_visible'][0]
        # elif 'keypoints_visible' in sample:
        #     sample['keypoints_visible'] = sample['keypoints_visible'][0]
        def pack_keypoints(s):
            if 'transformed_keypoints' in s:
                s['keypoints_'] = s['keypoints']
                s['keypoints'] = s.pop('transformed_keypoints')[0,:, :2]
            if 'transformed_keypoints_visible' in s:
                s['keypoints_visible'] = s['transformed_keypoints_visible'][0]
            elif 'keypoints_visible' in s:
                s['keypoints_visible'] = s['keypoints_visible'][0]
            return s

        sample = pack_keypoints(sample)

        if 'img' in sample:
            sample['image'] = sample.pop('img')
            sample['image'] = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB)
        else:
            H, W = sample['input_size']
            sample['image'] = np.zeros((H, W, 3), dtype=np.uint8)

        sample = self._transform_sample(sample, self.transform)
        sample['keypoints'] = np.array(sample['keypoints'])

        return sample

    def _get_dino_features(self, idx: int) -> np.ndarray:
        return np.array(self.attentions[str(idx)])

    def _get_sample(self, idx):

        data_info = self.get_data_info(idx)
        data_info['original_keypoints'] = data_info['keypoints'][0, :, :2].copy()

        data_info = self.pipeline(data_info)

        sample = data_info
        try:
            sample.pop('segmentation')
            sample.pop('raw_ann_info')
        except KeyError:
            pass
        try:
            sample.pop('flip_direction')
        except KeyError:
            pass

        H, W = sample['input_size']

        kp = sample['transformed_keypoints']
        # visible = sample['transformed_keypoints_visible'] > 0.5
        vkp = kp  #[visible]

        validx = (0 <= vkp[..., 0]) & (vkp[...,0] < W)
        validy = (0 <= vkp[..., 1]) & (vkp[...,1] < H)
        valid  = validy & validx

        if 'tranformed_keypoints_visible' in sample:
            sample['transformed_keypoints_visible'][~valid] = 0
        if 'keypoints_visible' in sample:
            sample['keypoints_visible'][~valid] = 0

        # assert np.all(validx)
        # assert np.all(validy)

        # create bodymaps
        # K, H, W = sample['heatmaps'].shape
        # from pose.landmarks.lmutils import create_bodymap
        # sample['heatmaps'] = create_bodymap(sample['image'], sample['keypoints'], sample['skeleton_links'],
        #                                     target_size=(W,H),
        #                                     keypoints_visible=sample['transformed_keypoints_visible'])

        sample['skeleton_links'] = np.array(sample['skeleton_links'])
        sample['skeleton_link_colors'] = np.array(sample['skeleton_link_colors'])

        # add precomputed DINO features
        if True:
            if self.attentions is not None:
                a = self._get_dino_features(sample['id'])
                if sample.get('flip', False):
                    a = a[:, :, ::-1]
                # a = self.dinovis.transform(a[np.newaxis]).transpose(0, 3, 1, 2)[0].astype(np.float32) / 255
                sample['attentions'] = a

        return sample

    def _transform_sample(self, sample, transform):

        # if not self.test_mode:
        #     dropout = alb.Compose([
        #         alb.CoarseDropout(
        #             min_width=8, min_height=8,
        #             max_width=64, max_height=64,
        #             min_holes=1,
        #             max_holes=6,
        #             p=0.25),
        #     ])
        #     sample['image'] = dropout(image=sample['image'])['image']

        sample = transform(**sample)
        # sample = self.norm(**sample)

        # try:
        #     sample['keypoints'] = np.array(sample['keypoints'])
        #     sample['landmarks'] = sample['keypoints'][:, :2]  # landmarks don't have size and angle
        # except KeyError:
        #     pass

        # self._check_all_masks(sample)
        return sample

    def get_data_info(self, idx: int) -> dict:
        """Get data info by index.

        Args:
            idx (int): Index of data info.

        Returns:
            dict: Data info.
        """
        data_info = super().get_data_info(idx)

        # Add metainfo items that are required in the pipeline and the model
        metainfo_keys = [
            'upper_body_ids', 'lower_body_ids', 'flip_pairs',
            'dataset_keypoint_weights', 'flip_indices', 'skeleton_links', 'skeleton_link_colors'
        ]

        for key in metainfo_keys:
            assert key not in data_info, (
                f'"{key}" is a reserved key for `metainfo`, but already '
                'exists in the `data_info`.')

            data_info[key] = deepcopy(self._metainfo[key])

        return data_info

    def load_data_list(self) -> List[dict]:
        """Load data list from COCO annotation file or person detection result
        file."""

        if self.bbox_file:
            data_list = self._load_detection_results()
        else:
            instance_list, image_list = self._load_annotations()

            if self.data_mode == 'topdown':
                data_list = self._get_topdown_data_infos(instance_list)
            else:
                data_list = self._get_bottomup_data_infos(
                    instance_list, image_list)

            print_log(f"Number of images/instances: {len(image_list)}/{len(instance_list)}", logger='current')
            # num_annotated_images = len(image_list) if not self.load_unlabeled else len(self.labeled_ids)
            # num_annotated_instances = sum([i['labeled'] for i in instance_list])
            # print_log(f"Number of annotated images/instances: {num_annotated_images}/{num_annotated_instances}", logger='current')

        return data_list

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in COCO format."""

        assert exists(self.ann_file), 'Annotation file does not exist'

        with get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        # set the metainfo about categories, which is a list of dict
        # and each dict contains the 'id', 'name', etc. about this category
        self._metainfo['CLASSES'] = self.coco.loadCats(self.coco.getCatIds())

        instance_list = []
        image_list = []

        all_img_ids = self.coco.getImgIds()

        self.img_ids = all_img_ids

        if self.labeled_ids is not None:
            if self.image_annotation_type == 'unlabeled':
                self.img_ids = [i for i in all_img_ids if i not in self.labeled_ids]
            if self.image_annotation_type == 'labeled':
                self.img_ids = self.labeled_ids
            if self.image_annotation_type is None or self.image_annotation_type == 'full':
                # load all images
                pass

        for img_id in self.img_ids:
            img = self.coco.loadImgs(img_id)[0]
            img.update({
                'img_id': img_id,
                'img_path': osp.join(self.data_prefix['img'], img['file_name']),
            })
            image_list.append(img)

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            for ann in self.coco.loadAnns(ann_ids):

                instance_info = self.parse_data_info(
                    dict(raw_ann_info=ann, raw_img_info=img))

                # skip invalid instance annotation.
                if not instance_info:
                    continue

                instance_info['labeled'] = 1
                if self.labeled_ids is not None and img_id not in self.labeled_ids:
                    # instance_info['keypoints'][:] = 0
                    # instance_info['keypoints_visible'][:] = 0
                    instance_info['labeled'] = 0

                instance_list.append(instance_info)

        return instance_list, image_list

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw COCO annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict | None: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        # filter invalid instance
        if 'bbox' not in ann or 'keypoints' not in ann:
            return None

        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(
            ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        if 'num_keypoints' in ann:
            num_keypoints = ann['num_keypoints']
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img['img_path'],
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'category_id': ann['category_id'],
            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            'raw_ann_info': copy.deepcopy(ann),
        }

        if 'crowdIndex' in img:
            data_info['crowd_index'] = img['crowdIndex']

        return data_info

    @staticmethod
    def _is_valid_instance(data_info: Dict) -> bool:
        """Check a data info is an instance with valid bbox and keypoint
        annotations."""
        # crowd annotation
        if 'iscrowd' in data_info and data_info['iscrowd']:
            return False
        # invalid keypoints
        if 'num_keypoints' in data_info and data_info['num_keypoints'] == 0:
            return False
        # invalid bbox
        if 'bbox' in data_info:
            bbox = data_info['bbox'][0]
            w, h = bbox[2:4] - bbox[:2]
            if w <= 0 or h <= 0:
                return False
        # invalid keypoints
        # if 'keypoints' in data_info:
        #     if np.max(data_info['keypoints']) <= 0:
        #         return False
        return True

    def _get_topdown_data_infos(self, instance_list: List[Dict]) -> List[Dict]:
        """Organize the data list in top-down mode."""
        # sanitize data samples
        data_list_tp = list(filter(self._is_valid_instance, instance_list))

        return data_list_tp

    def _get_bottomup_data_infos(self, instance_list: List[Dict],
                                 image_list: List[Dict]) -> List[Dict]:
        """Organize the data list in bottom-up mode."""

        # bottom-up data list
        data_list_bu = []

        used_img_ids = set()

        # group instances by img_id
        for img_id, data_infos in groupby(instance_list,
                                          lambda x: x['img_id']):
            used_img_ids.add(img_id)
            data_infos = list(data_infos)

            # image data
            img_path = data_infos[0]['img_path']
            data_info_bu = {
                'img_id': img_id,
                'img_path': img_path,
            }

            for key in data_infos[0].keys():
                if key not in data_info_bu:
                    seq = [d[key] for d in data_infos]
                    if isinstance(seq[0], np.ndarray):
                        seq = np.concatenate(seq, axis=0)
                    data_info_bu[key] = seq

            # The segmentation annotation of invalid objects will be used
            # to generate valid region mask in the pipeline.
            invalid_segs = []
            for data_info_invalid in filterfalse(self._is_valid_instance,
                                                 data_infos):
                if 'segmentation' in data_info_invalid:
                    invalid_segs.append(data_info_invalid['segmentation'])
            data_info_bu['invalid_segs'] = invalid_segs

            data_list_bu.append(data_info_bu)

        # add images without instance for evaluation
        if self.test_mode:
            for img_info in image_list:
                if img_info['img_id'] not in used_img_ids:
                    data_info_bu = {
                        'img_id': img_info['img_id'],
                        'img_path': img_info['img_path'],
                        'id': list(),
                        'raw_ann_info': None,
                    }
                    data_list_bu.append(data_info_bu)

        return data_list_bu

    def _load_detection_results(self) -> List[dict]:
        """Load data from detection results with dummy keypoint annotations."""

        assert exists(self.ann_file), 'Annotation file does not exist'
        assert exists(self.bbox_file), 'Bbox file does not exist'
        # load detection results
        det_results = load(self.bbox_file)
        assert is_list_of(det_results, dict)

        # load coco annotations to build image id-to-name index
        with get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        # set the metainfo about categories, which is a list of dict
        # and each dict contains the 'id', 'name', etc. about this category
        self._metainfo['CLASSES'] = self.coco.loadCats(self.coco.getCatIds())

        num_keypoints = self.metainfo['num_keypoints']
        data_list = []
        id_ = 0
        for det in det_results:
            # remove non-human instances
            if det['category_id'] != 1:
                continue

            img = self.coco.loadImgs(det['image_id'])[0]

            img_path = osp.join(self.data_prefix['img'], img['file_name'])
            bbox_xywh = np.array(
                det['bbox'][:4], dtype=np.float32).reshape(1, 4)
            bbox = bbox_xywh2xyxy(bbox_xywh)
            bbox_score = np.array(det['score'], dtype=np.float32).reshape(1)

            # use dummy keypoint location and visibility
            keypoints = np.zeros((1, num_keypoints, 2), dtype=np.float32)
            keypoints_visible = np.ones((1, num_keypoints), dtype=np.float32)

            data_list.append({
                'img_id': det['image_id'],
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'bbox_score': bbox_score,
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'id': id_,
            })

            id_ += 1

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg. Defaults return full
        ``data_list``.

        If 'bbox_score_thr` in filter_cfg, the annotation with bbox_score below
        the threshold `bbox_score_thr` will be filtered out.
        """

        data_list = self.data_list

        if self.filter_cfg is None:
            return data_list

        # filter out annotations with a bbox_score below the threshold
        if 'bbox_score_thr' in self.filter_cfg:

            if self.data_mode != 'topdown':
                raise ValueError(
                    f'{self.__class__.__name__} is set to {self.data_mode} '
                    'mode, while "bbox_score_thr" is only supported in '
                    'topdown mode.')

            thr = self.filter_cfg['bbox_score_thr']
            data_list = list(
                filterfalse(lambda ann: ann['bbox_score'] < thr, data_list))

        return data_list
