import copy

from .gfl_head import GFLHead
from .dual_head import DualNanoDetPlusHead
from .nanodet_head import NanoDetHead
from .nanodet_plus_head import NanoDetPlusHead
from .simple_conv_head import SimpleConvHead
from .stal_dual_head import STALDualNanoDetPlusHead
from .stal_nanodet_plus_head import STALNanoDetPlusHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    if name == "GFLHead":
        return GFLHead(**head_cfg)
    elif name == "NanoDetHead":
        return NanoDetHead(**head_cfg)
    elif name == "NanoDetPlusHead":
        return NanoDetPlusHead(**head_cfg)
    elif name == "DualNanoDetPlusHead":
        return DualNanoDetPlusHead(**head_cfg)
    elif name == "STALNanoDetPlusHead":
        return STALNanoDetPlusHead(**head_cfg)
    elif name == "STALDualNanoDetPlusHead":
        return STALDualNanoDetPlusHead(**head_cfg)
    elif name == "SimpleConvHead":
        return SimpleConvHead(**head_cfg)
    else:
        raise NotImplementedError
