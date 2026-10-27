"""Pure validation shared by config parsing and ranking LR control."""

import math


def validate_validation_scheduler_options(options, initial_lr):
    """Validate the same settings for config loading and optimizer construction."""
    if options is None:
        return None
    if not isinstance(options, dict) or set(options) != {"factor", "patience", "min_lr"}:
        raise ValueError("validation_lr_scheduler needs factor, patience and min_lr")
    factor, patience, min_lr = options["factor"], options["patience"], options["min_lr"]
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or not 0 < factor < 1:
        raise ValueError("Scheduler factor must be in (0, 1)")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
        raise ValueError("Scheduler patience must be a nonnegative integer")
    if isinstance(min_lr, bool) or not isinstance(min_lr, (int, float)) or not math.isfinite(min_lr) or min_lr <= 0:
        raise ValueError("Scheduler min_lr must be finite and positive")
    if isinstance(initial_lr, bool) or not isinstance(initial_lr, (int, float)) or not math.isfinite(initial_lr) or initial_lr <= 0:
        raise ValueError("Scheduler initial LR must be finite and positive")
    if min_lr > initial_lr:
        raise ValueError("Scheduler min_lr exceeds initial learning rate")
    return options


