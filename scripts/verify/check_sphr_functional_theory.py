"""Exact SPHR functional-design checks and finite counterexamples, without training."""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/".tmp/sympy114"))
import sympy as s


def check():
    if s.__version__ != "1.14.0":
        raise RuntimeError("Use isolated SymPy 1.14.0")
    groups, formulas = [], {}
    lam = s.symbols("lambda", real=True)
    f = (lam**2+lam+2)/4
    precision = 1/f**2
    tangent = 1+s.Rational(3, 2)*(1-lam)
    difference = (1-lam)**2*(3*lam**3+7*lam**2+16*lam+12)/(2*(lam**2+lam+2)**2)
    assert s.factor(precision-tangent-difference) == 0
    assert precision.subs(lam, 1) == tangent.subs(lam, 1) == 1
    assert s.diff(precision, lam).subs(lam, 1) == s.diff(tangent, lam) == -s.Rational(3, 2)
    assert precision.subs(lam, 0) == 4 and tangent.subs(lam, 0) == s.Rational(5, 2)
    formulas["spectral_precision_remainder"] = s.sstr(difference)
    groups.append("exact tangent relaxation of inverse-filter precision; remainder nonnegative for lambda in [0,1]")

    interaction = s.Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    du, di = s.diag(2, 2, 1), s.diag(1, 2, 2)
    half = s.diag(1, s.sqrt(2), s.sqrt(2))
    transition = di.inv()*interaction.T*du.inv()*interaction
    symmetric = half*transition*half.inv()
    assert symmetric == symmetric.T
    operator = s.eye(3)/2+transition/4+transition**2/4
    whitened = s.eye(3)/2+symmetric/4+symmetric**2/4
    assert half*operator*half.inv() == whitened
    assert (operator.T*di-di*operator) == s.zeros(3)
    groups.append("degree-weighted similarity transform exactly symmetrizes the actual random-walk filter on an unequal-degree graph")

    averaging = s.ones(2)/2
    two_item_filter = s.eye(2)/2+averaging/4+averaging**2/4
    desired = s.Matrix([1, -1])
    assert two_item_filter.inv()*desired == s.Matrix([2, -2])
    assert two_item_filter*s.Matrix([1, -1]) == desired/2
    groups.append("invertibility does not remove row-ball restrictions: feasible target vectors [1,-1] need infeasible raw vectors [2,-2] in a two-item averaging graph")

    x = s.Matrix(s.symbols("x0:3", real=True))
    graph_precision = di-interaction.T*du.inv()*interaction
    sos = 0
    for row in interaction.tolist():
        support = [i for i, value in enumerate(row) if value]
        sos += sum((x[i]-x[j])**2 for i in support for j in support)/(2*len(support))
    assert s.expand((x.T*graph_precision*x)[0]-sos) == 0
    assert s.simplify(operator.inv().T*di*operator.inv()-half*whitened.inv()**2*half) == s.zeros(3)
    groups.append("explicit co-history Dirichlet penalty is a sum of squares; old degree penalty transforms to inverse-filter-squared precision")

    alpha = s.symbols("alpha", positive=True)
    resolvent = (s.eye(3)+alpha*(s.eye(3)-transition)).inv()
    assert s.simplify(resolvent*s.ones(3, 1)-s.ones(3, 1)) == s.zeros(3, 1)
    assert s.simplify((s.eye(3)+alpha*(s.eye(3)-transition))*resolvent) == s.eye(3)
    groups.append("weighted quadratic graph proximal step satisfies its normal equations and preserves the all-ones vector in the finite fixture")

    c, m, temperature = s.symbols("c m T", positive=True)
    logistic = lambda margin: s.log(1+s.exp(-margin/temperature))
    derivative = -2*c*m/(temperature*(1+s.exp(c**2*m/temperature)))
    assert s.simplify(s.diff(logistic(c**2*m), c)-derivative) == 0
    formulas["radial_derivative"] = s.sstr(derivative)
    groups.append("joint radial scaling of residual/source/target reduces correctly signed logistic loss without changing their rankings")

    beta, beta0, eta = s.symbols("beta beta0 eta", positive=True)
    margins = s.symbols("m0:2", real=True)
    gain_loss = sum(logistic(beta*z) for z in margins)/2+eta*(beta-beta0)**2/2
    gain_curvature = eta+sum(z**2*s.exp(beta*z/temperature)/(
        temperature**2*(1+s.exp(beta*z/temperature))**2) for z in margins)/2
    assert s.simplify(s.diff(gain_loss, beta, 2)-gain_curvature) == 0
    groups.append("positive gain with a positive quadratic prior has a strongly convex scalar conditional objective; beta/T is the identifiable logit scale")

    omega, residual, history, eta_w = s.symbols("omega residual history eta_w", real=True)
    gate_loss = logistic(residual+omega*history)+eta_w*omega**2/2
    gate_curvature = eta_w+history**2*s.exp((residual+omega*history)/temperature)/(
        temperature**2*(1+s.exp((residual+omega*history)/temperature))**2)
    assert s.simplify(s.diff(gate_loss, omega, 2)-gate_curvature) == 0
    groups.append("history gate is a convex scalar conditional problem when embeddings/residuals are frozen and eta_w is nonnegative")

    cr, ch = s.symbols("cr ch", real=True)
    feature1, feature2 = s.Matrix([1, 2]), s.Matrix([-1, 1])
    coefficients = s.Matrix([cr, ch])
    exponents = [s.exp(-vector.dot(coefficients)) for vector in (feature1, feature2)]
    normalizer = 1+sum(exponents)/2
    probabilities = [1/normalizer, *(value/(2*normalizer) for value in exponents)]
    p0, p1, p2 = probabilities
    mean_feature = p1*feature1+p2*feature2
    covariance_matrix = p1*feature1*feature1.T+p2*feature2*feature2.T-mean_feature*mean_feature.T
    assert s.simplify(s.hessian(s.log(normalizer), (cr, ch))-covariance_matrix) == s.zeros(2)
    squares_matrix = p0*p1*feature1*feature1.T+p0*p2*feature2*feature2.T
    squares_matrix += p1*p2*(feature1-feature2)*(feature1-feature2).T
    assert s.simplify(covariance_matrix-squares_matrix) == s.zeros(2)
    groups.append("two-coefficient frozen history/residual log-sum-exp Hessian equals a PSD covariance; positive anchor L2 makes the conditional calibration strongly convex")

    theta = s.symbols("theta", real=True)
    user = s.Matrix([theta, s.Rational(1, 2)])
    positive, easy, hard = s.zeros(2, 1), s.Matrix([-1, 0]), s.Matrix([1, -1])/s.sqrt(2)
    me, mh = (user.dot(positive-easy), user.dot(positive-hard))
    epsilon = s.Rational(1, 500)
    toy = (1-epsilon)*logistic(me)+epsilon*logistic(mh)
    start, stop = s.Rational(1, 4), s.Rational(3, 4)
    before, after = [toy.subs({theta: v, temperature: 1}) for v in (start, stop)]
    regularizer = user.dot(user)/20
    objective_before, objective_after = [(toy.subs(temperature, 1)+regularizer).subs(theta, v) for v in (start, stop)]
    assert user.dot(user).subs(theta, stop) == s.Rational(13, 16)
    assert easy.dot(easy) == hard.dot(hard) == 1
    assert mh.subs(theta, start) > 0 and mh.subs(theta, stop) < 0
    assert after.evalf(40) < before.evalf(40)
    assert objective_after.evalf(40) < objective_before.evalf(40)
    toy_values = {"N": 10000, "K": 20, "theta_before": str(start), "theta_after": str(stop),
                  "bpr_before": float(before.evalf(40)), "bpr_after": float(after.evalf(40)),
                  "bpr_plus_user_penalty_before": float(objective_before.evalf(40)),
                  "bpr_plus_user_penalty_after": float(objective_after.evalf(40)),
                  "rank_before": 1, "rank_after": 21,
                  "topk_logistic_before": float(logistic(mh).subs({theta: start, temperature: 1}).evalf(40)),
                  "topk_logistic_after": float(logistic(mh).subs({theta: stop, temperature: 1}).evalf(40))}
    groups.append("feasible SPHR-MF two-dimensional counterexample: BPR and BPR+lambda_r=.1 penalty decline while the positive rank changes from1 to21")

    losses = [s.Rational(i, 3) for i in (1, 2, 3, 4)]
    k = 2
    values = [threshold+sum(s.Max(loss-threshold, 0) for loss in losses)/k for threshold in losses]
    assert min(values) == sum(sorted(losses, reverse=True)[:k])/k == s.Rational(7, 6)
    for signs in itertools.product((-1, 1), repeat=2):
        scores = [s.Integer(signs[0]), s.Integer(signs[1])]
        bad = any(margin < 0 for margin in scores)
        full_loss = s.log(1+sum(s.exp(-margin) for margin in scores))
        if bad:
            assert full_loss.evalf(30) >= s.log(2).evalf(30)
    groups.append("finite top-K/CVaR threshold identity and complete-list miss-bound examples; the general inequalities are proved separately in the report")

    rho = s.symbols("rho", positive=True)
    ell, threshold = s.symbols("ell threshold", real=True)
    smooth_hinge = rho*s.log(1+s.exp((ell-threshold)/rho))
    soft = 1/(1+s.exp((threshold-ell)/rho))
    assert s.simplify(s.diff(smooth_hinge, threshold)+soft) == 0
    assert s.simplify(s.diff(smooth_hinge, threshold, 2)-soft*(1-soft)/rho) == 0
    groups.append("smooth-CVaR threshold derivatives yield sum(sigmoid)=K; finite minimizer requires 0<K<N")

    a, v = s.symbols("a v", real=True)
    ridge, tie = s.symbols("ridge tie", positive=True)
    coupled = s.log(1+s.exp(-a*v))+ridge*(a*a+v*v)/2+tie*(a-v)**2/2
    hessian = s.hessian(coupled, (a, v)).subs({a: 0, v: 0})
    assert hessian*s.Matrix([1, 1]) == (ridge-s.Rational(1, 2))*s.Matrix([1, 1])
    assert hessian*s.Matrix([1, -1]) == (ridge+s.Rational(1, 2)+2*tie)*s.Matrix([1, -1])
    groups.append("soft-tied joint problem remains nonconvex: one Hessian eigenvalue is ridge-1/2 regardless of tie strength")

    y = s.symbols("y", real=True)
    trap = s.Max(theta, y)+(theta**2+y**2)/2
    assert trap.subs({theta: 0, y: 0}) == 0
    assert trap.subs({theta: -s.Rational(1, 4), y: -s.Rational(1, 4)}) == -s.Rational(3, 16)
    groups.append("nonsmooth coupled max admits a coordinatewise minimum at0 with a lower joint direction; a smooth-block theorem cannot be applied to exact top-K blindly")

    error_r, error_h, covariance = s.symbols("error_r error_h covariance", real=True)
    mse = (1-omega)**2*error_r+omega**2*error_h+2*omega*(1-omega)*covariance
    optimum = (error_r-covariance)/(error_r+error_h-2*covariance)
    assert s.factor(s.diff(mse, omega).subs(omega, optimum)) == 0
    assert s.expand(s.diff(mse, omega, 2)-2*(error_r+error_h-2*covariance)) == 0
    groups.append("correlated-error MSE gate formula holds when error-difference variance is positive; it is not a ranking-optimality theorem")

    return {"status": "pass", "sympy": s.__version__, "groups": len(groups), "checks": groups,
            "formulas": formulas, "feasible_bpr_counterexample": toy_values,
            "new_training": False, "test_labels_read": False,
            "scope": "exact identities plus finite 40-digit evaluations; general inequalities and convergence assumptions are stated separately"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
