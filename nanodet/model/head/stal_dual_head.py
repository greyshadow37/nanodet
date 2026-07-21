"""NMS-free dual NanoDet head with STAL supervision."""

import math

import numpy as np
import torch
import torch.nn as nn

from nanodet.util import distance2bbox

from ..loss.progressive_balance import ProgressiveLossBalancer
from .stal_nanodet_plus_head import STALNanoDetPlusHead


class STALDualNanoDetPlusHead(nn.Module):
    """Dense STAL o2m training branch plus true one-to-one STAL o2o branch.

    Only ``o2o`` is emitted for ONNX export and used by ``post_process``.
    Post-processing performs thresholding and top-k selection only; it never
    invokes NMS.
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
        stal_cfg=dict(small_target_threshold=8.0, reference_size=16.0),
        inference_head="o2o",
        score_thr=0.05,
        max_det=100,
        balancer_cfg=dict(),
        **kwargs
    ):
        super().__init__()
        if inference_head != "o2o":
            raise ValueError("STALDualNanoDetPlusHead supports o2o inference only")
        common = dict(
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
        self.o2m_head = STALNanoDetPlusHead(
            **common, stal_cfg={**dict(stal_cfg), "matching_mode": "one_to_many"}
        )
        self.o2o_head = STALNanoDetPlusHead(
            **common, stal_cfg={**dict(stal_cfg), "matching_mode": "one_to_one"}
        )
        self.inference_head = inference_head
        self.score_thr = score_thr
        self.max_det = max_det
        self.balancer = ProgressiveLossBalancer(**balancer_cfg)

    def forward(self, feats):
        if torch.onnx.is_in_onnx_export():
            return self.o2o_head(feats)
        return {"o2m": self.o2m_head(feats), "o2o": self.o2o_head(feats)}

    def loss(self, preds, gt_meta, epoch=None, aux_preds=None, **kwargs):
        o2m_loss, o2m_states = self.o2m_head.loss(
            preds["o2m"], gt_meta, aux_preds=aux_preds
        )
        o2o_loss, o2o_states = self.o2o_head.loss(
            preds["o2o"], gt_meta, aux_preds=aux_preds
        )
        o2m_weight, o2o_weight = self.balancer.weights(epoch)
        states = {"o2m_weight": o2m_loss.new_tensor(o2m_weight), "o2o_weight": o2o_loss.new_tensor(o2o_weight)}
        states.update({"o2m_" + key: value for key, value in o2m_states.items()})
        states.update({"o2o_" + key: value for key, value in o2o_states.items()})
        return o2m_weight * o2m_loss + o2o_weight * o2o_loss, states

    def post_process(self, preds, meta):
        """Decode the o2o output without NMS."""
        preds = preds["o2o"] if isinstance(preds, dict) else preds
        head = self.o2o_head
        cls_scores, bbox_preds = preds.split([head.num_classes, 4 * (head.reg_max + 1)], dim=-1)
        height, width = meta["img"].shape[2:]
        priors = torch.cat(
            [
                head.get_single_level_center_priors(
                    cls_scores.size(0),
                    (math.ceil(height / stride), math.ceil(width / stride)),
                    stride,
                    torch.float32,
                    cls_scores.device,
                )
                for stride in head.strides
            ],
            dim=1,
        )
        distances = head.distribution_project(bbox_preds) * priors[..., 2, None]
        boxes = distance2bbox(priors[..., :2], distances, max_shape=(height, width))
        scores = cls_scores.sigmoid()
        img_ids = meta["img_info"]["id"]
        heights, widths, matrices = meta["img_info"]["height"], meta["img_info"]["width"], meta["warp_matrix"]
        results = {}
        for index in range(cls_scores.size(0)):
            confidence, labels = scores[index].max(dim=1)
            keep = confidence > self.score_thr
            confidence, labels, selected_boxes = confidence[keep], labels[keep], boxes[index][keep]
            if confidence.numel() > self.max_det:
                confidence, order = confidence.topk(self.max_det)
                labels, selected_boxes = labels[order], selected_boxes[order]
            detections = torch.cat([selected_boxes, confidence[:, None]], dim=1).detach().cpu().numpy()
            image_id = int(img_ids[index]) if isinstance(img_ids, torch.Tensor) else img_ids[index]
            image_height = int(heights[index]) if isinstance(heights, torch.Tensor) else heights[index]
            image_width = int(widths[index]) if isinstance(widths, torch.Tensor) else widths[index]
            matrix = matrices[index] if isinstance(matrices, list) else matrices[index]
            if detections.size:
                from ...data.transform.warp import warp_boxes

                detections[:, :4] = warp_boxes(detections[:, :4], np.linalg.inv(matrix), image_width, image_height)
            class_ids = labels.detach().cpu().numpy()
            results[image_id] = {
                class_id: detections[class_ids == class_id].astype(np.float32).tolist()
                for class_id in range(head.num_classes)
            }
        return results
