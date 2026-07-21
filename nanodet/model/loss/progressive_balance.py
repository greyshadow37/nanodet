class ProgressiveLossBalancer:
    """Epoch-aware linear progression between one-to-many and one-to-one loss.

    This is a pragmatic implementation of progressive balancing:
    - early epochs favor the one-to-many branch for dense supervision
    - later epochs gradually shift weight toward the one-to-one branch
    """

    def __init__(
        self,
        warmup_epochs=0,
        transition_epochs=50,
        o2m_start=1.0,
        o2m_end=0.25,
        o2o_start=0.25,
        o2o_end=1.0,
        power=1.0,
    ):
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = max(1, transition_epochs)
        self.o2m_start = o2m_start
        self.o2m_end = o2m_end
        self.o2o_start = o2o_start
        self.o2o_end = o2o_end
        self.power = power

    def _progress(self, epoch):
        if epoch is None:
            return 1.0
        if epoch < self.warmup_epochs:
            return 0.0
        t = (epoch - self.warmup_epochs) / float(self.transition_epochs)
        t = max(0.0, min(1.0, t))
        return t**self.power

    def weights(self, epoch=None):
        p = self._progress(epoch)
        o2m = self.o2m_start + (self.o2m_end - self.o2m_start) * p
        o2o = self.o2o_start + (self.o2o_end - self.o2o_start) * p
        return o2m, o2o
