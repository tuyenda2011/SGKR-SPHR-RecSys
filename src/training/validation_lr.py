"""Optional ranking-monitored learning-rate control with serializable state."""

from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.validation_lr_options import validate_validation_scheduler_options


def build_validation_scheduler(optimizer, options):
    """Disable by default; reject settings that can mask a malformed experiment."""
    if options is None:
        return None
    validate_validation_scheduler_options(options, min(group["lr"] for group in optimizer.param_groups))
    # The same strict improvement definition as EarlyStopping. No smoothed or
    # test-derived metric, cooldown, or hidden relative threshold.
    return ReduceLROnPlateau(optimizer, mode="max", factor=options["factor"], patience=options["patience"],
                             threshold=0.0, threshold_mode="abs", min_lr=options["min_lr"], eps=0.0)
