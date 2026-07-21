from .one_stage_detector import OneStageDetector


class DualHeadDetector(OneStageDetector):
    def forward_train(self, gt_meta):
        feat = self.backbone(gt_meta["img"])
        fpn_feat = self.fpn(feat) if hasattr(self, "fpn") else feat
        head_out = self.head(fpn_feat)
        loss, loss_states = self.head.loss(head_out, gt_meta, epoch=self.epoch)
        return head_out, loss, loss_states
