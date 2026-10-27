"""Exact boundary-penalty identities; no recommendation training or data access."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/".tmp/sympy114"))
import sympy as s


def check():
    if s.__version__ != "1.14.0":
        raise RuntimeError("Use SymPy 1.14.0")
    xi, eta = s.symbols("xi eta", nonnegative=True)
    x, y = s.symbols("x y", real=True)
    normalizer = s.diag(1/s.sqrt(5), 1/s.sqrt(3))
    inside = normalizer*s.Matrix([[2, -2], [-2, 2]])*normalizer
    boundary = s.diag(s.Rational(3, 5), s.Rational(1, 3))
    principal = s.eye(2)-normalizer*s.Matrix([[0, 2], [2, 0]])*normalizer
    assert principal == inside+boundary
    vector = s.Matrix([x, y])
    sos = 2*(x/s.sqrt(5)-y/s.sqrt(3))**2+xi*(3*x*x/5+y*y/3)
    assert s.expand((vector.T*(inside+xi*boundary)*vector)[0]-sos) == 0
    assert (inside+xi*boundary).subs(xi, 1) == principal
    assert (inside+xi*boundary).subs(xi, 0) == inside
    assert s.diff((vector.T*(inside+xi*boundary)*vector)[0], xi) == 3*x*x/5+y*y/3
    determinant = s.factor((inside+xi*boundary+eta*s.eye(2)).det())
    assert s.expand(determinant-(eta**2+16*eta/15+14*eta*xi/15+3*xi**2/15+8*xi/15)) == 0
    return {"status": "pass", "sympy": s.__version__, "groups": 3,
            "checks": ["principal block = internal + boundary for an unequal-degree graph",
                       "quadratic energy is an exact sum of squares for xi>=0",
                       "eta>0 gives positive determinant and positive diagonals in the fixture; general SPD follows from positive ridge plus PSD graph energy"],
            "determinant": str(determinant), "new_training": False,
            "scope": "exact identities on a three-node graph fixture; general PSD proof accompanies implementation documentation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
