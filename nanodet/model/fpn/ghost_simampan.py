from .ghost_pan import GhostPAN


class GhostSimAMPAN(GhostPAN):
    """GhostPAN with SimAM enabled by default."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_simam", True)
        super().__init__(*args, **kwargs)
