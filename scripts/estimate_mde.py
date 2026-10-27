"""Plan NDCG@20 sensitivity from existing validation predictions; never fit or score a model."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow  # noqa: F401 -- load Arrow DLLs before Torch on Windows
import scipy
from scipy.stats import norm

from scripts.compare_val_runs import (
    half_mask,
    per_user_metrics,
    validate_run_arrays,
    verify_run_protocol,
)
from scripts.validation_io import edge_hash, load_validation_inputs
from src.data.quality_report import warm_cohort


def gate_mde(se, comparisons, effect_floor, *, alpha=.05, power=.8):
    if (not math.isfinite(se) or se < 0 or not isinstance(comparisons, int) or isinstance(comparisons, bool)
            or comparisons < 1 or not math.isfinite(effect_floor) or effect_floor < 0
            or not 0 < alpha < power < 1):
        raise ValueError("Invalid MDE planning controls")
    critical = float(norm.isf(alpha/comparisons))
    z_power = float(norm.ppf(power))
    effective_cutoff = max(critical*se, effect_floor)
    return {"comparisons": comparisons, "alpha_first_holm": alpha/comparisons,
            "z_critical": critical, "effect_floor": effect_floor,
            "mde_holm": (critical+z_power)*se,
            "mde_with_effect_floor": effective_cutoff+z_power*se,
            "threshold_below_mde": effect_floor < (critical+z_power)*se,
            "power_at_prespecified_floor": float(norm.sf((effective_cutoff-effect_floor)/se)) if se else None}


def pair_sensitivity(differences, n_selection, n_confirmation, core_ndcg, *, alpha=.05, power=.8):
    differences = np.asarray(differences, dtype=np.float64)
    if (differences.ndim != 1 or len(differences) < 2 or not np.isfinite(differences).all()
            or min(n_selection, n_confirmation) < 1 or n_selection+n_confirmation != len(differences)
            or not math.isfinite(core_ndcg) or not 0 <= core_ndcg <= 1):
        raise ValueError("Expected finite paired differences and two nonempty validation halves")
    sd = float(differences.std(ddof=1))
    se = sd/math.sqrt(n_selection)
    gates = {name: gate_mde(se, count, floor, alpha=alpha, power=power)
             for name, count, floor in (("B", 4, .001), ("B_supplement", 3, .001), ("C", 8, .02*core_ndcg))}
    return {"n_users": len(differences), "n_selection": n_selection, "n_confirmation": n_confirmation,
            "sd_diff": sd, "se_all": sd/math.sqrt(len(differences)), "se_selection": se,
            "se_confirmation": sd/math.sqrt(n_confirmation), "degenerate_proxy": sd == 0., "gates": gates,
            "extra_comparison_sensitivity": {
                "B_with_H2": gate_mde(se, 5, .001, alpha=alpha, power=power),
                "C_with_H8": gate_mde(se, 9, .02*core_ndcg, alpha=alpha, power=power)}}


def markdown(report):
    rows = ["Ước tính độ nhạy từ dự đoán có sẵn; không phải lực quan sát từ hiệu ứng đã đo.", "",
            f"| Cặp | Cổng | Số so sánh | SE selection | MDE {report['target_power']*100:g}% Holm | Ngưỡng | Ngưỡng < MDE |",
            "|---|---|---:|---:|---:|---:|---|"]
    for pair in report["pairs"]:
        for name, gate in pair["gates"].items():
            rows.append(f"| {pair['label']} | {name} | {gate['comparisons']} | {pair['se_selection']:.8f} | "
                        f"{gate['mde_holm']:.8f} | {gate['effect_floor']:.8f} | "
                        + ("có" if gate["threshold_below_mde"] else "không") + " |")
    rows.extend(["", report["limits"]])
    return "\n".join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, nargs=2, action="append", required=True, metavar=("RUN_A", "RUN_B"))
    parser.add_argument("--core-reference", type=Path, required=True, help="Existing core predictions for the C effect floor")
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--power", type=float, default=.8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.alpha < args.power < 1:
        raise ValueError("Expected 0 < alpha < power < 1")
    started = time.perf_counter()
    bundle, train, validation, stats = load_validation_inputs(args.bundle, ROOT)
    warm, _ = warm_cohort(train, validation)
    if warm.empty or warm.u_idx.duplicated().any():
        raise ValueError("Expected exactly one warm validation target per user")
    warm = warm.sort_values("u_idx")
    users, targets = (warm[key].to_numpy(dtype=np.int64) for key in ("u_idx", "i_idx"))
    n_selection = int(half_mask(users, "selection").sum())
    n_confirmation = len(users)-n_selection
    identity = {"bundle_id": bundle.build_id, "density": 1., "density_seed": 42,
                "retained_sha256": edge_hash(train), "mask_sha256": edge_hash(train), "cohort_sha256": edge_hash(warm)}
    metrics_cache = {}

    def existing_metrics(path):
        path = path.resolve()
        if path not in metrics_cache:
            verify_run_protocol(path, identity)
            predictions = validate_run_arrays(path, users, train, train, stats["num_items"])
            metrics_cache[path] = per_user_metrics(predictions, targets)["NDCG@20"]
        return metrics_cache[path]

    core_ndcg = float(existing_metrics(args.core_reference).mean())
    report = {"validation_only": True, "model_fit_called": False, "model_score_called": False,
              "core_reference": str(args.core_reference.resolve()), "core_ndcg20": core_ndcg,
              "alpha": args.alpha, "target_power": args.power, "protocol": identity,
              "formula": "SE=sd(diff,ddof=1)/sqrt(n_half); MDE=(z(1-alpha/m)+z(power))*SE",
              "method": "xấp xỉ chuẩn cho trung bình hiệu theo cặp user; dùng bước Holm đầu tiên bảo thủ",
              "limits": "Giả định user độc lập có điều kiện và phương sai proxy giữ nguyên; arm tương lai có thể nhiễu hơn. "
                        "MDE Holm chưa gồm ngưỡng hiệu ứng, nhóm later hay confirmation; xem mde_with_effect_floor. "
                        "Không chứng minh lực 80% cho toàn quy trình bootstrap/chọn arm. Không tự đổi ngưỡng hay số arm.",
              "thresholds": "B-bổ sung kế thừa +0.001 của B; C tạm lấy 2% core bỏ kernel có sẵn, phải cập nhật khi core cuối được khóa",
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}, "pairs": []}
    for index, (a, b) in enumerate(args.pair, 1):
        result = pair_sensitivity(existing_metrics(b)-existing_metrics(a), n_selection, n_confirmation,
                                  core_ndcg, alpha=args.alpha, power=args.power)
        report["pairs"].append({"label": f"P{index}", "a": str(a.resolve()), "b": str(b.resolve()), **result})
    report["seconds"] = time.perf_counter()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(markdown(report))
    return report


if __name__ == "__main__":
    main()
