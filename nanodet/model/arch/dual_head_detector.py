import copy

import torch

from ..head import build_head
from .one_stage_detector import OneStageDetector


class DualHeadDetector(OneStageDetector):
    """Dual-head detector with optional Assign Guidance Module (AGM).

    When ``aux_head`` is configured, an auxiliary FPN/head pair provides
    detached predictions that guide STAL+DSLA label assignment for both the
    one-to-many and one-to-one training branches.
    """

    def __init__(
        self,
        backbone,
        fpn,
        head,
        aux_head=None,
        detach_epoch=0,
    ):
        super().__init__(backbone_cfg=backbone, fpn_cfg=fpn, head_cfg=head)
        self.detach_epoch = detach_epoch
        if aux_head is not None:
            self.aux_fpn = copy.deepcopy(self.fpn)
            self.aux_head = build_head(aux_head)
        else:
            self.aux_fpn = None
            self.aux_head = None

    def _dual_fpn_features(self, feat, fpn_feat):
        if self.epoch >= self.detach_epoch:
            aux_fpn_feat = self.aux_fpn([level.detach() for level in feat])
            return (
                torch.cat([level.detach(), aux_level], dim=1)
                for level, aux_level in zip(fpn_feat, aux_fpn_feat)
            )
        aux_fpn_feat = self.aux_fpn(feat)
        return (
            torch.cat([level, aux_level], dim=1)
            for level, aux_level in zip(fpn_feat, aux_fpn_feat)
        )

    def forward_train(self, gt_meta):
        img = gt_meta["img"]
        feat = self.backbone(img)
        fpn_feat = self.fpn(feat)
        head_out = self.head(fpn_feat)

        aux_preds = None
        if self.aux_head is not None:
            dual_fpn_feat = self._dual_fpn_features(feat, fpn_feat)
            aux_preds = self.aux_head(dual_fpn_feat)

        loss, loss_states = self.head.loss(
            head_out, gt_meta, epoch=self.epoch, aux_preds=aux_preds
        )
        return head_out, loss, loss_states
