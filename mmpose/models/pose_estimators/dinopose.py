from itertools import zip_longest
from typing import Dict, Optional, Tuple, Union
from collections import OrderedDict
from mmengine.optim import OptimWrapper
import torch
import numpy as np
import kornia

import time
import cv2
import os
from torch import Tensor
import h5py

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator
from mmpose.utils.tensor_utils import to_numpy
from mmpose.evaluation.functional import pose_pck_accuracy
from mmengine.structures import PixelData
from mmpose.models.utils.tta import flip_heatmaps

import cv2


total_curr_iter = 0

from mmengine.model import BaseModule, constant_init
import torch.nn as nn


@MODELS.register_module()
class DinoPoseEstimator(BasePoseEstimator):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 head_hr: OptConfigType = None,
                 dino_encoder: ConfigType = None,
                 dino_neck: OptConfigType = None,
                 dino_decoder: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None,
                 distill=False):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)

        if dino_encoder is not None:
            self.dino_encoder = MODELS.build(dino_encoder)
        if dino_neck is not None:
            self.dino_neck = MODELS.build(dino_neck)
        if dino_decoder is not None:
            self.dino_decoder = MODELS.build(dino_decoder)
        if head_hr is not None:
            self.head_hr = MODELS.build(head_hr)

        self.dino_hdf5 = None
        self.dino_features = {}

        self.batch_idx = 0

        self.distill = distill

    def get_dino_inputs(self, data_samples) -> Tensor | None:
        if 'dino' not in data_samples[0]:
            return None
        return torch.stack([s.dino for s in data_samples])

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """

        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            _data = self.data_preprocessor(data, True)
            losses, _ = self.forward(**_data, train=True)  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        _data = self.data_preprocessor(data, True)
        losses, results = self.forward(**_data, train=False)  # type: ignore
        _, log_vars = self.parse_losses(losses)  # type: ignore
        return [results, log_vars]

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        _data = self.data_preprocessor(data, True)
        losses, results = self.forward(**_data, train=False)  # type: ignore
        _, log_vars = self.parse_losses(losses)  # type: ignore
        return [results, log_vars]

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        pass

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        pass

    def forward_backbone(self, backbone, inputs, train):
        with torch.set_grad_enabled(train):
            if not train and self.test_cfg.get('flip_test', False):
                _feats = backbone(inputs)
                _feats_flip = backbone(inputs.flip(-1))
                feats = [_feats, _feats_flip]
            else:
                feats = backbone(inputs)
        return feats

    def forward_neck(self, neck, feats, train):
        with torch.set_grad_enabled(train):
            if not train and self.test_cfg.get('flip_test', False):
                _feats = neck(feats[0])
                _feats_flip = neck(feats[1])
                feats = [_feats, _feats_flip]
            else:
                feats = neck(feats)
        return feats

    def forward_head(self, head, feats, data_samples, train):
        if not train and self.test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _pred_heatmaps = head.forward(_feats)
            _pred_heatmaps_flip = head.forward(_feats_flip)
            _pred_heatmaps_flip = flip_heatmaps(
                _pred_heatmaps_flip,
                flip_mode=self.test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=self.test_cfg.get('shift_heatmap', False))
            pred_heatmaps = (_pred_heatmaps + _pred_heatmaps_flip) * 0.5
        else:
            pred_heatmaps = head.forward(feats)
        return pred_heatmaps


    def merge_feats(self, feats1, feats2, train):
        if not train and self.test_cfg.get('flip_test', False):
            feats_merged = [(feats1[0][0] + feats2[0][0]) / 2.0, (feats1[1][0] + feats2[1][0]) / 2.0]
        else:
            feats_merged = [(feats1[0] + feats2[0]) / 2.0]
        return feats_merged

    def forward(self, inputs: Tensor, data_samples: SampleList, train=False) -> tuple:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """

        # def count_parameters(m):
        #     return sum(p.numel() for p in m.parameters() if p.requires_grad)
        # print(count_parameters(self.backbone))
        # print(count_parameters(self.student_head_attn))
        # exit()

        inputs_dino = self.get_dino_inputs(data_samples)

        masks_rgb = np.stack([s.mask for s in data_samples])

        masks = torch.tensor(masks_rgb, device='cuda').unsqueeze(1)
        masks = torch.nn.functional.interpolate(masks.float(), size=(64, 64)).bool()

        if inputs_dino is not None:
            masks_dino = inputs_dino.sum(axis=1, keepdims=True) != 0
            masks = masks_dino & masks
            m_recon = masks.repeat(1, inputs_dino.shape[1], 1, 1)
        else:
            m_recon = None

        losses = dict()

        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in data_samples])
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in data_samples])

        results = None
        freeze_backbone = False

        # predict backbone
        with torch.set_grad_enabled((not freeze_backbone or self.distill) and train):
            feats = self.forward_backbone(self.backbone, inputs, train)
            feats = self.forward_neck(self.neck, feats, train)

        if self.distill:
            ft_ = feats
            if not train:
                ft_ = feats[0]
            dino_recon = self.dino_decoder(ft_)
            if inputs_dino.shape[1] == 12:
                # DINOv1
                loss_recon_dino = torch.nn.functional.mse_loss(dino_recon[m_recon], inputs_dino[m_recon], reduction='sum')
            else:
                # DINOv2
                loss_recon_dino = torch.nn.functional.mse_loss(dino_recon[m_recon], inputs_dino[m_recon]) * 0.1
            losses.update(loss_recon_dino=loss_recon_dino)
        else:
            # detach features
            if train and freeze_backbone:
                feats = feats.detach()

            # predict keypoints directly from dino features
            pred_heatmaps = self.forward_head(self.head, feats, data_samples, train)

            loss_kpt_student = self.head.loss_module(pred_heatmaps, gt_heatmaps, keypoint_weights) * 100.0
            losses.update(loss_kpt_student=loss_kpt_student)

            preds = self.head.decode(pred_heatmaps)
            pred_fields = [PixelData(heatmaps=hm) for hm in pred_heatmaps.detach().cpu()]

            # calculate accuracy
            if self.train_cfg.get('compute_acc', True):
                _, avg_acc, _ = pose_pck_accuracy(
                    output=to_numpy(pred_heatmaps),
                    target=to_numpy(gt_heatmaps),
                    mask=to_numpy(keypoint_weights) > 0)

                acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
                losses.update(acc_pose=acc_pose)

            results = self.add_pred_to_datasample(preds, pred_fields, data_samples)

        self.batch_idx += 1
        return losses, results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            input_center = data_sample.metainfo['input_center']
            input_scale = data_sample.metainfo['input_scale']
            input_size = data_sample.metainfo['input_size']

            pred_instances.keypoints[..., :2] = \
                pred_instances.keypoints[..., :2] / input_size * input_scale \
                + input_center - 0.5 * input_scale
            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = \
                    pred_instances.keypoint_scores

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
