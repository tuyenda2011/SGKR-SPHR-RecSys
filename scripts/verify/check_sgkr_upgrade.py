"""Toy-only independent correctness checks for SGKR support penalty controls."""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd  # noqa: F401  # Import before scipy on this Windows runtime.

from scripts.run_sgkr import branch_diagnostics, parser, run
from src.models.sgkr import SGKR, SGKRConfig


def explicit_design(model, item):
    """Dense construction independent of cached Gram/design_rows implementation."""
    raw = model.interactions.toarray()
    raw[:, item] = 0
    norms = np.sqrt(np.maximum(raw.sum(axis=1), 1))
    context = raw / norms[:, None]
    return np.column_stack((context[:, model.neighborhoods[item]], np.ones(model.num_users)))


def check_fit(config, users, items):
    model = SGKR(config).fit(users, items, 5, 6)
    raw = model.interactions.toarray()
    all_users = np.arange(5)
    seen_replacement_changed = False
    maximum_gap = 0.
    for item in range(6):
        if model.counts[item] == 0:
            assert model.bias[item] == model.prior(item)[-1]
            continue
        design = explicit_design(model, item)
        np.testing.assert_allclose(design, model.design_rows(all_users, item), atol=1e-14)
        observed = raw[:, item]
        weights = 1 + (config.positive_weight - 1) * observed
        penalty = model.precision(item)
        neighbors = model.neighborhoods[item]
        support = model.counts / np.maximum(model.counts + config.alpha, 1)
        source_scale = support[neighbors] ** -config.adaptive_power if config.adaptive_regularization else np.ones(len(neighbors))
        if config.adaptive_cap is not None:
            source_scale = np.minimum(source_scale, config.adaptive_cap)
        expected_penalty = np.diag(np.concatenate((config.eta_linear * source_scale, [config.eta_bias])))
        expected_penalty[:len(neighbors), :len(neighbors)] += (
            config.graph_weight * model.laplacian[neighbors][:, neighbors].toarray())
        np.testing.assert_allclose(penalty, model.num_users * expected_penalty, rtol=1e-14, atol=1e-14)
        assert np.linalg.eigvalsh(penalty).min() > 0
        expected = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                                   design.T @ (weights * observed) + penalty @ model.prior(item))
        actual = np.concatenate((model.linear_coefficients[neighbors, item].toarray().ravel(),
                                 [model.bias[item]]))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        gap = float(np.max(np.abs(design @ expected - model.score(all_users)[:, item])))
        maximum_gap = max(maximum_gap, gap)
        unexcluded = model.design_rows(all_users, item, exclude_target=False)
        seen_replacement_changed |= not np.allclose(design, unexcluded)
    assert seen_replacement_changed
    assert model.fit_report["prior_fallback_items"] == 1
    for exact in (False, True):
        parts = model.score_parts(all_users, np.arange(6), exclude_seen=exact)
        np.testing.assert_allclose(sum(parts.values()), model.score(all_users, exclude_seen=exact), atol=1e-12)
        # Candidate/user reordering and repeated users retain exact semantics.
        subset = model.score_parts(np.array([3, 0, 3]), np.array([4, 0, 2]), exclude_seen=exact)
        np.testing.assert_allclose(sum(subset.values()),
                                   model.score(np.array([3, 0, 3]), exclude_seen=exact)[:, [4, 0, 2]], atol=1e-12)
    return model, maximum_gap


def main():
    users = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4])
    items = np.array([0, 1, 2, 0, 3, 1, 4, 2, 3, 4])
    # Full graph block and support shrinkage tied to alpha: the penalty forms derived below.
    base = SGKRConfig(neighbors=4, alpha=10, correction_batch_size=1, boundary_weight=1., support_alpha=None)
    checks, maximum_gap = 0, 0.
    models = []
    for power in (0., 1., 2.):
        for cap in (None, 1., 10.):
            config = replace(base, adaptive_power=power, adaptive_cap=cap)
            model, gap = check_fit(config, users, items)
            models.append(model)
            maximum_gap = max(maximum_gap, gap)
            scales = model.support_multipliers(np.arange(6))
            expected = np.ones(6)
            expected[:5] = model.support[:5] ** -power
            if cap is not None:
                expected = np.minimum(expected, cap)
                assert scales.max() <= cap * (1 + 1e-14)
            np.testing.assert_allclose(scales, expected, rtol=1e-14)
            checks += 1
    for config in (replace(base, graph_weight=0),
                   replace(base, adaptive_regularization=False),
                   replace(base, alpha=0, adaptive_power=2., adaptive_cap=10.)):
        check_fit(config, users, items)
        checks += 1
    legacy = models[6]  # Power 2, no cap: historical behavior.
    for item in range(5):
        n = legacy.neighborhoods[item]
        p = legacy.precision(item) / legacy.num_users
        expected = np.diag(base.eta_linear * legacy.support[n] ** -2)
        expected += base.graph_weight * legacy.laplacian[n][:, n].toarray()
        np.testing.assert_array_equal(p[:len(n), :len(n)], expected)
    legacy_dict = asdict(base)
    legacy_dict.pop("adaptive_power")
    legacy_dict.pop("adaptive_cap")
    assert SGKRConfig(**legacy_dict) == base
    with tempfile.TemporaryDirectory() as temporary:
        for index, model in enumerate((legacy, models[-1])):
            path = Path(temporary) / f"model{index}.npz"
            model.save(path)
            restored = SGKR.load(path)
            assert restored.config == model.config
            np.testing.assert_array_equal(restored.score(np.arange(5)), model.score(np.arange(5)))
        path = Path(temporary) / "legacy.npz"
        original = Path(temporary) / "model0.npz"
        with np.load(original, allow_pickle=False) as state:
            payload = {key: state[key].copy() for key in state.files}
        payload["config"] = np.array(json.dumps(legacy_dict))
        np.savez_compressed(path, **payload)
        restored = SGKR.load(path)
        np.testing.assert_array_equal(restored.score(np.arange(5)), legacy.score(np.arange(5)))
    # Changing the penalty must not change neighbors or support.
    for model in models:
        for left, right in zip(model.neighborhoods, legacy.neighborhoods):
            np.testing.assert_array_equal(left, right)
    evaluator = SimpleNamespace(eval_users=[0], ground_truth=[{3}])
    diagnostic = branch_diagnostics(legacy, evaluator, np.array([[4, 3]]), 1)
    assert diagnostic["users"][0]["margins"][0]["negative"] == 4
    assert branch_diagnostics(legacy, evaluator, np.array([[4, 3]]), 0)["users"] == []
    for invalid in ({"adaptive_power": -1}, {"adaptive_power": float("inf")},
                    {"adaptive_cap": .9}, {"adaptive_cap": True}, {"adaptive_cap": float("nan")}):
        try:
            SGKRConfig(**invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid config: {invalid}")
    run(parser().parse_args(["--dry-run", "--adaptive-power", "1", "--adaptive-cap", "10"]))
    print(json.dumps({"status": "passed", "toy_fits": checks, "dense_score_max_gap": maximum_gap,
                      "checks": ["independent_dense_target_excluded_normal_equations", "positive_row_replacement",
                                 "source_and_target_scales_spd", "unchanged_neighborhoods",
                                 "branch_sum_seen_and_unseen", "new_and_legacy_checkpoint_roundtrip",
                                 "invalid_controls", "bounded_branch_margin_diagnostic", "runner_dry_run"],
                      "real_data_read": False, "quality_improvement_claim": False}, indent=2))


if __name__ == "__main__":
    main()
