"""Exact small-instance and scalar checks for the SPHR proposal; no model training."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".tmp/sympy114"))
import sympy as s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if s.__version__ != "1.14.0":
        raise RuntimeError("Use SymPy 1.14.0")

    interaction = s.Matrix([[1, 1, 0], [0, 1, 1], [1, 0, 0]])
    user_degree = s.diag(2, 2, 1)
    item_degree = s.diag(2, 2, 1)
    root = s.diag(s.sqrt(2), s.sqrt(2), 1)
    transition = item_degree.inv() * interaction.T * user_degree.inv() * interaction
    symmetric = s.simplify(root * transition * root.inv())
    ones = s.ones(3, 1)
    assert transition * ones == ones
    stationary = s.Matrix([[s.Rational(2, 5), s.Rational(2, 5), s.Rational(1, 5)]])
    assert stationary * transition == stationary
    assert item_degree * transition == transition.T * item_degree
    assert symmetric == symmetric.T
    eigenvalues = symmetric.eigenvals()
    assert all(value.is_nonnegative is True and (1 - value).is_nonnegative is True for value in eigenvalues)

    filt = s.eye(3) / 2 + transition / 4 + transition ** 2 / 4
    normalized_filter = s.eye(3) / 2 + symmetric / 4 + symmetric ** 2 / 4
    assert filt * ones == ones
    assert all(entry >= 0 for entry in filt)
    assert s.simplify(root * filt * root.inv() - normalized_filter) == s.zeros(3)
    assert filt.det() > 0
    lam = s.symbols("eigenvalue", nonnegative=True)
    gain = s.Rational(1, 2) + lam / 4 + lam ** 2 / 4
    assert s.diff(gain, lam).is_positive is True
    assert gain.subs(lam, 0) == s.Rational(1, 2)
    assert gain.subs(lam, 1) == 1

    base = s.Matrix([[1, 0], [0, 1], [-1, 1]])
    filtered = filt * base
    spectral_energy = s.trace((root * filtered).T * normalized_filter.inv() ** 2 * root * filtered)
    base_energy = s.trace(base.T * item_degree * base)
    assert s.simplify(spectral_energy - base_energy) == 0
    # Stationary Jensen contraction makes a fully trainable identity filter a degeneracy.
    assert s.trace(filtered.T * item_degree * filtered) <= base_energy

    degree, tau = s.symbols("degree tau", positive=True)
    observed, prior_mean, latent = s.symbols("observed prior_mean latent", real=True)
    objective = degree * (observed - latent) ** 2 + tau * (latent - prior_mean) ** 2
    posterior_mean = (degree * observed + tau * prior_mean) / (degree + tau)
    assert s.simplify(s.diff(objective, latent).subs(latent, posterior_mean)) == 0
    rho = degree / (degree + tau)
    assert s.simplify(posterior_mean - prior_mean - rho * (observed - prior_mean)) == 0
    effective_penalty = 1 / rho ** 2
    assert s.factor(s.diff(effective_penalty, degree)) == -2 * tau * (degree + tau) / degree ** 3
    assert rho.subs({degree: 1, tau: 4}) == s.Rational(1, 5)
    singleton_gates = [rho.subs({degree: 1, tau: value}) for value in (1, 8)]
    singleton_penalties = [effective_penalty.subs({degree: 1, tau: value}) for value in (1, 8)]
    assert singleton_gates == [s.Rational(1, 2), s.Rational(1, 9)]
    assert singleton_penalties == [4, 81]

    margin = s.symbols("margin", real=True)
    loss = s.log(1 + s.exp(-margin))
    first = s.diff(loss, margin)
    second = s.diff(loss, margin, 2)
    assert s.simplify(first + 1 / (1 + s.exp(margin))) == 0
    assert s.simplify(second - s.exp(margin) / (1 + s.exp(margin)) ** 2) == 0
    assert s.simplify(s.Rational(1, 4) - second
                      - (s.exp(margin) - 1) ** 2 / (4 * (1 + s.exp(margin)) ** 2)) == 0
    assert loss.subs(margin, 0) == s.log(2)

    mixture, left, right = s.symbols("mixture left right", real=True)
    assert s.expand(mixture * left ** 2 + (1 - mixture) * right ** 2
                    - (mixture * left + (1 - mixture) * right) ** 2
                    - mixture * (1 - mixture) * (left - right) ** 2) == 0
    # Singleton context is zero but personal residual produces item gradients.
    p = s.Matrix([[1, 2]]) / 5
    q = s.Matrix(3, 2, s.symbols("q0:6", real=True))
    v = filt * q
    singleton_margin = (p * (v[0, :] - v[1, :]).T)[0]
    singleton_loss = s.log(1 + s.exp(-singleton_margin))
    derivative = singleton_loss.diff(q[0, 0])
    expected = -(filt[0, 0] - filt[1, 0]) * p[0, 0] / (1 + s.exp(singleton_margin))
    assert s.simplify(derivative - expected) == 0
    assert s.simplify(derivative.subs(dict.fromkeys(q, 0))) != 0

    report = {
        "status": "pass", "sympy_version": s.__version__,
        "checks": ["row-stochastic item transition and stationary distribution",
                   "reversibility and similarity to a symmetric PSD contraction on an exact example",
                   "fixed filter invertibility, nonnegative rows and spectral gain polynomial",
                   "degree-weighted regularizer expressed as an exact spectral prior",
                   "Gaussian shrinkage stationary point and residual penalty monotonicity",
                   "BPR derivative, curvature bound and loss at a tie",
                   "convex mixture Jensen identity for norm bounds",
                   "nonzero singleton item gradient on an exact example",
                   "tau search endpoints imply singleton gates 1/2 to 1/9 and penalty multipliers 4 to 81"],
        "example_transition": [[str(value) for value in transition.row(row)] for row in range(3)],
        "example_spectrum": [str(value) for value in eigenvalues],
        "filter_gain": str(gain), "effective_residual_penalty": str(effective_penalty),
        "singleton_search_gate_endpoints": [str(value) for value in singleton_gates],
        "singleton_search_penalty_endpoints": [str(value) for value in singleton_penalties],
        "learnable_parameters_at_dim64": (124895 + 44843) * 64,
        "limits": ["General proofs and their assumptions are in the design document.",
                   "Scalar identities and one exact matrix example do not prove arbitrary-size results.",
                   "No claim of Adam convergence to a global optimum or prevention of all overfitting.",
                   "No SPHR training or empirical superiority claim."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
