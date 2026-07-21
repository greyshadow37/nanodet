"""NanoDet-Plus head using Small-Target-Aware Label Assignment (STAL)."""

from .assigner.stal_assigner import SmallTargetAwareDynamicSoftLabelAssigner
from .nanodet_plus_head import NanoDetPlusHead


class STALNanoDetPlusHead(NanoDetPlusHead):
    """NanoDetPlusHead whose DSLA candidate stage is small-target aware."""

    def __init__(self, *args, stal_cfg=None, **kwargs):
        assigner_cfg = kwargs.pop("assigner_cfg", None) or {"topk": 13}
        stal_cfg = stal_cfg or {}
        super().__init__(*args, assigner_cfg=assigner_cfg, **kwargs)
        self.assigner = SmallTargetAwareDynamicSoftLabelAssigner(
            **dict(assigner_cfg), **dict(stal_cfg)
        )
