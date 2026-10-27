"""Euclidean projected gradient with a full-objective descent check.

The gradient mapping is evaluated at the pre-step point. Small gradient mapping
certifies approximate constrained stationarity, not ranking/generalization.
"""
import math

import torch
from src.utils.source_identity import source_identity

SOURCE_SHA256 = source_identity(__file__)


def _row_radii(values, radius):
    radii = torch.as_tensor(radius, dtype=values.dtype, device=values.device)
    if values.ndim != 2 or radii.ndim > 1 or (radii.ndim == 1 and len(radii) != len(values)):
        raise ValueError("Radius must be a scalar or one value per row")
    if not bool(torch.isfinite(radii).all()) or bool((radii <= 0).any()):
        raise ValueError("Radii must be positive and finite")
    return radii.expand(len(values))[:, None]


@torch.no_grad()
def diagonal_ball_projection(values, metric, radius, iterations=40):
    """Project rows onto an L2 ball in the positive diagonal metric H.

    KKT gives x_i=H_i*z_i/(H_i+mu), mu>=0. Bisection finds the unique active
    multiplier. Normalize H rowwise (the minimizer is invariant to that scale).
    """
    if values.ndim != 2 or metric.shape != values.shape:
        raise ValueError("Expected equal row matrices")
    radii = _row_radii(values, radius)
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(metric).all()) or bool((metric <= 0).any()):
        raise ValueError("Projection inputs must be finite and metric strictly positive")
    outside = values.norm(dim=1) > radii[:, 0]
    result = values.clone()
    if not bool(outside.any()):
        return result
    selected, h = values[outside], metric[outside]
    selected_radii = radii[outside]
    h = h/h.max(dim=1, keepdim=True).values
    ratio = selected.norm(dim=1, keepdim=True)/selected_radii
    low = h.min(dim=1, keepdim=True).values*(ratio-1).clamp_min(0)
    high = ratio
    for _ in range(iterations):
        # Geometric brackets resolve small multipliers for anisotropic Adam metrics.
        mid = torch.where(low > 0, low.sqrt()*high.sqrt(), (low+high)/2)
        candidate = selected*h/(h+mid)
        infeasible = candidate.norm(dim=1, keepdim=True) > selected_radii
        low = torch.where(infeasible, mid, low)
        high = torch.where(infeasible, high, mid)
    projected = selected*h/(h+high)
    # Roundoff-only feasibility correction after using the feasible bracket end.
    projected *= (selected_radii/projected.norm(dim=1, keepdim=True).clamp_min(1e-30)).clamp_max(1)
    result[outside] = projected
    return result


@torch.no_grad()
def project_adam_parameter(parameter, radius, optimizer):
    """Use the same diagonal metric as the immediately preceding Adam step."""
    group = next(group for group in optimizer.param_groups if any(p is parameter for p in group["params"]))
    state = optimizer.state[parameter]
    if "exp_avg_sq" not in state:
        raise ValueError("Adam must update this parameter before metric projection")
    variance = state["max_exp_avg_sq"] if group.get("amsgrad", False) else state["exp_avg_sq"]
    radii = _row_radii(parameter, radius)
    outside = parameter.norm(dim=1) > radii[:, 0]
    fraction = float(outside.float().mean())
    if fraction == 0:
        return fraction
    step = float(state["step"])
    metric = variance[outside].sqrt()/math.sqrt(1-group["betas"][1]**step)+group["eps"]
    parameter[outside] = diagonal_ball_projection(parameter[outside], metric, radii[outside, 0])
    return fraction


def projected_step(parameter, project, objective, step_size, max_backtracks=30, strong_convexity=None, metric=None):
    """Full-objective PGD; project must minimize distance in the supplied metric.

    A diagonal metric is valid with coordinate bounds/free coordinates. Coupled
    constraints require their corresponding metric projection, not Euclidean projection.
    When requesting the optional strong-convexity certificate, project must ALSO
    be the Euclidean projection (coordinate clamps satisfy both requirements).
    """
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("Projected-gradient step size must be positive and finite")
    if metric is not None and (metric.shape != parameter.shape or not bool(torch.isfinite(metric).all())
                               or bool((metric <= 0).any())):
        raise ValueError("Metric must match parameters and be finite and strictly positive")
    before = objective(backward=True)
    origin = parameter.detach().clone()
    gradient = parameter.grad.detach().clone()
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("Nonfinite full-objective gradient")
    gradient_norm = float(gradient.double().norm())
    inverse_direction = gradient if metric is None else gradient/metric
    metric_gradient_norm = gradient_norm if metric is None else float((gradient.double().square()/metric).sum().sqrt())
    certificate = {}
    if strong_convexity is not None and strong_convexity > 0:
        with torch.no_grad():
            parameter.copy_(origin-gradient/strong_convexity)
            project()
            direction = (parameter-origin).double()
            gap_bound = -(gradient.double()*direction).sum()-strong_convexity*direction.square().sum()/2
            parameter.copy_(origin)
        certificate = {"cached_objective_gap_upper_bound_before": max(0., float(gap_bound)),
                       "strong_convexity": strong_convexity}
    for backtracks in range(max_backtracks+1):
        with torch.no_grad():
            parameter.copy_(origin-step_size*inverse_direction)
            project()
            displacement = parameter-origin
            squared_distance = displacement.double().square()
            if metric is not None:
                squared_distance = squared_distance*metric
            mapping_norm = float(squared_distance.sum().sqrt())/step_size
            if backtracks == 0:
                reference_mapping_norm = mapping_norm
            bound = (before["objective"] + float((gradient.double()*displacement.double()).sum())
                     + float(squared_distance.sum())/(2*step_size))
        after = objective(backward=False)
        if not all(math.isfinite(value) for value in after.values()):
            with torch.no_grad():
                parameter.copy_(origin)
            raise FloatingPointError("Nonfinite trial objective")
        # Absolute float64 summation tolerance; never scale it by parameter count.
        tolerance = 1e-10*max(1., abs(before["objective"]))
        if after["objective"] <= min(bound, before["objective"])+tolerance:
            return after, {**certificate, "step_size": step_size, "backtracks": backtracks,
                           "gradient_norm": gradient_norm, "gradient_mapping_norm": reference_mapping_norm,
                           "metric_gradient_norm": metric_gradient_norm,
                           "gradient_mapping_metric": "euclidean" if metric is None else "fixed_diagonal",
                           "projection_ratio": reference_mapping_norm/max(metric_gradient_norm, 1e-30),
                           "stationarity_at": "before_step", "solver_step_accepted": True}
        step_size *= .5
    with torch.no_grad():
        parameter.copy_(origin)
    return before, {**certificate, "step_size": step_size*2, "backtracks": max_backtracks,
                    "gradient_norm": gradient_norm, "gradient_mapping_norm": reference_mapping_norm,
                    "metric_gradient_norm": metric_gradient_norm,
                    "gradient_mapping_metric": "euclidean" if metric is None else "fixed_diagonal",
                    "projection_ratio": reference_mapping_norm/max(metric_gradient_norm, 1e-30),
                    "stationarity_at": "before_step", "solver_step_accepted": False}
