"""Exact identities supporting the SGKR proposal, not a baseline-win proof."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".tmp" / "sympy114"))
import sympy as s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert s.__version__ == "1.14.0"
    a, b = s.symbols("a b", positive=True)
    n = s.symbols("n", integer=True, positive=True)
    total = a+b
    mean_c = n*a/total
    variance_c = n*a*b*(total+n)/(total**2*(total+1))
    expected_posterior_variance = ((a+mean_c)*(b+n-mean_c)-variance_c) / ((total+n)**2*(total+n+1))
    bayes_risk = a*b/(total*(total+1)*(n+total))
    mle_risk = a*b/(n*total*(total+1))
    assert s.simplify(expected_posterior_variance-bayes_risk) == 0
    assert s.simplify(bayes_risk/mle_risk-n/(n+total)) == 0
    # Shrinkage is not uniformly better at every fixed true probability.
    assert (s.Rational(2, 3)-1)**2 == s.Rational(1, 9)

    probability = s.symbols("probability", nonnegative=True)
    weight = s.symbols("positive_weight", positive=True)
    prediction = s.symbols("prediction", real=True)
    optimum = weight*probability/(1+(weight-1)*probability)
    risk = weight*probability*(1-prediction)**2+(1-probability)*prediction**2
    assert s.simplify(s.diff(risk, prediction).subs(prediction, optimum)) == 0
    assert s.simplify(risk-risk.subs(prediction, optimum)
                      -(1+(weight-1)*probability)*(prediction-optimum)**2) == 0
    monotone = s.factor(s.diff(optimum, probability))
    assert s.simplify(monotone-weight/(1+(weight-1)*probability)**2) == 0

    w12, w23 = s.symbols("w12 w23", positive=True)
    z1, z2, z3 = s.symbols("z1 z2 z3", real=True)
    laplacian = s.Matrix([[w12, -w12, 0], [-w12, w12+w23, -w23], [0, -w23, w23]])
    z = s.Matrix([z1, z2, z3])
    assert s.expand((z.T*laplacian*z)[0]-w12*(z1-z2)**2-w23*(z2-z3)**2) == 0

    design = s.Matrix([[1, 0], [1, 1], [0, 1]])
    confidence = s.diag(2, 3, 1)
    precision = s.Matrix([[2, -1], [-1, 2]])
    target = s.Matrix([1, 0, 1])
    prior = s.Matrix([s.Rational(1, 5), 0])
    hessian = design.T*confidence*design+precision
    theta = hessian.inv()*(design.T*confidence*target+precision*prior)
    assert hessian*theta-design.T*confidence*target-precision*prior == s.zeros(2, 1)
    assert hessian.is_positive_definite
    baseline = s.Matrix([[1, 1], [1, 0], [0, 1]])
    excluded = s.Matrix([[0, 1], [1, 0], [1, 1]])
    weighted = s.diag(7, 1, 7)
    corrected = baseline.T*baseline
    for index in (0, 2):
        positive = excluded[index, :].T
        original = baseline[index, :].T
        corrected += 7*positive*positive.T-original*original.T
    assert corrected == excluded.T*weighted*excluded
    assert excluded.T*weighted*target == 7*(excluded[0, :].T+excluded[2, :].T)
    singular_value = s.symbols("singular_value", positive=True)
    gain = singular_value**2/(1+singular_value**2)
    assert s.simplify(1-gain-1/(1+singular_value**2)) == 0
    assert gain.is_positive
    delta_p, delta_discount = s.symbols("delta_p delta_discount", positive=True)
    low_p, low_discount = s.symbols("low_p low_discount", nonnegative=True)
    ordered = (low_p+delta_p)*(low_discount+delta_discount)+low_p*low_discount
    reversed_order = (low_p+delta_p)*low_discount+low_p*(low_discount+delta_discount)
    assert s.expand(ordered-reversed_order) == delta_p*delta_discount

    report = {"status": "pass", "sympy_version": s.__version__,
              "beta_binomial_bayes_risk": str(bayes_risk),
              "risk_ratio_to_empirical_proportion": str(n/(n+total)),
              "fixed_probability_counterexample": "p=1,n=1,a=b=1: posterior risk1/9, empirical risk0",
              "weighted_square_optimum": str(optimum), "monotone_derivative": str(monotone),
              "laplacian_roughness_identity": "w12*(z1-z2)^2+w23*(z2-z3)^2",
              "exact_quadratic_solution": [str(value) for value in theta],
              "hat_matrix_spectral_gain": str(gain),
              "checks": ["Beta-Binomial integrated squared-risk shrinkage",
                         "weighted squared-loss optimum and excess-risk identity",
                         "positive class weight preserves ideal probability ordering",
                         "graph Laplacian quadratic-form identity",
                         "exact weighted regularized normal equations on a rational example",
                         "target-excluded positive-row cache corrections on a rational example",
                         "expected single-target DCG exchange gain",
                         "spectral gains between zero and one"],
              "limits": ["General matrix proofs and assumptions are in the theory document.",
                         "Risk reduction is integrated under a specified prior, not uniform over fixed p.",
                         "Convexity and spectral contraction condition on fixed features, graph and confidence.",
                         "Reconstruction probabilities need not equal chronological next-item probabilities.",
                         "No empirical SGKR training or proof of superiority over the three baselines."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
