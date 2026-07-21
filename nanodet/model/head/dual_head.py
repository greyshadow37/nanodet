import copy

import math

import numpy as np
import torch
import torch.nn as nn

from nanodet.util import distance2bbox

from .nanodet_plus_head import NanoDetPlusHead


class DualNanoDetPlusHead(nn.Module):
    """YOLOv10-style dual head for NanoDet.

    The one-to-many branch provides dense supervision during training.
    The one-to-one branch is used for low-latency end-to-end inference.
    """

    is_dual_head = True

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32],
        conv_type="DWConv",
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="LeakyReLU",
        assigner_cfg=dict(topk=13),
        inference_head="o2o",
        balancer_cfg=dict(
            warmup_epochs=0,
            transition_epochs=50,
            o2m_start=1.0,
            o2m_end=0.25,
            o2o_start=0.25,
            o2o_end=1.0,
            power=1.0,
        ),
        **kwargs
    ):
        super().__init__()
        self.o2m_head = NanoDetPlusHead(
            num_classes=num_classes,
            loss=loss,
            input_channel=input_channel,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs,
            kernel_size=kernel_size,
            strides=strides,
            conv_type=conv_type,
            norm_cfg=norm_cfg,
            reg_max=reg_max,
            activation=activation,
            assigner_cfg=assigner_cfg,
        )
        self.o2o_head = copy.deepcopy(self.o2m_head)
        self.inference_head = inference_head
        from ..loss.progressive_balance import ProgressiveLossBalancer

        self.balancer = ProgressiveLossBalancer(**balancer_cfg)

    def forward(self, feats):
        return {
            "o2m": self.o2m_head(feats),
            "o2o": self.o2o_head(feats),
        }

    def loss(self, preds, gt_meta, epoch=None, **kwargs):
        o2m_loss, o2m_states = self.o2m_head.loss(preds["o2m"], gt_meta)
        o2o_loss, o2o_states = self.o2o_head.loss(preds["o2o"], gt_meta)
        o2m_w, o2o_w = self.balancer.weights(epoch)

        total_loss = o2m_w * o2m_loss + o2o_w * o2o_loss
        loss_states = {"o2m_weight": o2m_w, "o2o_weight": o2o_w}

        for k, v in o2m_states.items():
            loss_states[f"o2m_{k}"] = v
        for k, v in o2o_states.items():
            loss_states[f"o2o_{k}"] = v
        return total_loss, loss_states

    def post_process(self, preds, meta):
        head = self.o2o_head if self.inference_head == "o2o" else self.o2m_head
        preds = preds[self.inference_head]
        cls_scores, bbox_preds = preds.split(
            [head.num_classes, 4 * (head.reg_max + 1)], dim=-1
        )

        device = cls_scores.device
        batch_size = cls_scores.shape[0]
        input_height, input_width = meta["img"].shape[2:]
        input_shape = (input_height, input_width)
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width / stride))
            for stride in head.strides
        ]
        mlvl_center_priors = [
            head.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(head.strides)
        ]

        center_priors = torch.cat(mlvl_center_priors, dim=1)
        dis_preds = head.distribution_project(bbox_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_scores.sigmoid()

        result_list = []
        score_thr = 0.05
        max_det = 100
        for i in range(batch_size):
            score_map = scores[i]
            bbox_map = bboxes[i]
            conf, labels = score_map.max(dim=1)
            keep = conf > score_thr
            if keep.any():
                conf = conf[keep]
                labels = labels[keep]
                bbox_map = bbox_map[keep]
                topk = min(max_det, conf.size(0))
                conf, order = conf.topk(topk)
                labels = labels[order]
                bbox_map = bbox_map[order]
                det_bboxes = torch.cat([bbox_map, conf[:, None]], dim=1)
            else:
                det_bboxes = bbox_map.new_zeros((0, 5))
                labels = bbox_map.new_zeros((0,), dtype=torch.long)

            det_result = {}
            img_height = (
                meta["img_info"]["height"][i]
                if isinstance(meta["img_info"]["height"], (list, tuple))
                else meta["img_info"]["height"]
            )
            img_width = (
                meta["img_info"]["width"][i]
                if isinstance(meta["img_info"]["width"], (list, tuple))
                else meta["img_info"]["width"]
            )
            warp_matrix = (
                meta["warp_matrix"][i]
                if isinstance(meta["warp_matrix"], list)
                else meta["warp_matrix"]
            )
            det_bboxes = det_bboxes.detach().cpu().numpy()
            if det_bboxes.size > 0:
                from ...data.transform.warp import warp_boxes

                det_bboxes[:, :4] = warp_boxes(
                    det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
                )
            classes = labels.detach().cpu().numpy()
            for cls_id in range(head.num_classes):
                inds = classes == cls_id
                det_result[cls_id] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            result_list.append(det_result)

        return result_list
