"""Exact checks for the Cholesky EASE solver in scripts/run_ease.py; not a ranking-performance proof.

1. EASE closed form: with P = (G + lambda I)^-1, B = I - P diag(1/diag P) has a zero diagonal and
   satisfies the KKT stationarity of  min ||X - XB||^2 + lambda ||B||^2  s.t. diag(B) = 0, i.e. the
   off-diagonal entries of (G + lambda I) B - G vanish. So only P and its diagonal are needed.
2. Cholesky inverse: for A = U^T U (U upper triangular), A^-1 = U^-1 U^-T, the product LAPACK potri
   forms in place (upper triangle). A is symmetric positive definite for any Gram G and lambda > 0,
   so potrf + potri replaces the general LU inverse.
"""
import argparse
import json
from pathlib import Path

import sympy as sp

N = 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    regularization = sp.symbols('lambda', positive=True)
    # A generic Gram matrix G = F^T F (symmetric positive semidefinite) of 3 items.
    factors = sp.Matrix(N, N, sp.symbols(f'f:{N * N}', real=True))
    gram = factors.T * factors
    system = gram + regularization * sp.eye(N)
    inverse = system.inv()
    closed_form = sp.eye(N) - inverse * sp.diag(*[1 / inverse[i, i] for i in range(N)])
    assert all(sp.simplify(closed_form[i, i]) == 0 for i in range(N))
    stationarity = sp.simplify(system * closed_form - gram)
    assert all(stationarity[i, j] == 0 for i in range(N) for j in range(N) if i != j)
    # Equivalent column scaling used in code: B = -P / diag(P) column-wise, then the diagonal set to 0.
    scaled = sp.Matrix(N, N, lambda i, j: 0 if i == j else -inverse[i, j] / inverse[j, j])
    assert sp.simplify(scaled - closed_form) == sp.zeros(N, N)

    upper = sp.Matrix(N, N, lambda i, j: sp.Symbol(f'u{i}{j}', positive=(i == j)) if j >= i else 0)
    spd = upper.T * upper
    assert sp.simplify(upper.inv() * upper.inv().T - spd.inv()) == sp.zeros(N, N)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'sympy_version': sp.__version__, 'status': 'pass', 'items': N,
              'ease_closed_form': 'B = I - P diag(1/diag P): zero diagonal, KKT off-diagonal stationarity',
              'column_scaling_equivalence': 'B_ij = -P_ij / P_jj (i != j)',
              'cholesky_inverse': 'A = U^T U  =>  A^-1 = U^-1 U^-T'}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
