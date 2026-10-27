"""Exact ranking identities and counterexamples; never fits a recommender."""
import argparse
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
    theta = s.Matrix(s.symbols("t0:2", real=True))
    feature = s.Matrix(s.symbols("a0:2", real=True))
    temperature, ridge = s.symbols("T lambda", positive=True)
    margin = theta.dot(feature)
    loss = s.log(1+s.exp(-margin/temperature))
    gradient = -feature/(temperature*(1+s.exp(margin/temperature)))
    curvature = s.exp(margin/temperature)/(temperature**2*(1+s.exp(margin/temperature))**2)
    assert s.simplify(s.Matrix([s.diff(loss,t) for t in theta])-gradient) == s.zeros(2, 1)
    assert s.simplify(s.hessian(loss, tuple(theta))-curvature*feature*feature.T) == s.zeros(2)
    checks = ["BPR gradient/Hessian for affine margins, exact symbolic parameters and T>0"]

    features = [s.Matrix([1, -2]), s.Matrix([-1, 1])]
    terms = [s.exp(-theta.dot(a)) for a in features]
    denominator = 1+sum(terms)/2
    probabilities = [v/(2*denominator) for v in terms]
    mean = sum((p*a for p,a in zip(probabilities,features)), s.zeros(2,1))
    covariance = sum((p*a*a.T for p,a in zip(probabilities,features)),s.zeros(2))-mean*mean.T
    assert s.simplify(s.hessian(s.log(denominator), tuple(theta))-covariance) == s.zeros(2)
    checks.append("listwise log-sum-exp Hessian equals a covariance fixture; general PSD argument is in report")

    laplacian = s.Matrix([[1,-1],[-1,1]])
    precision = ridge*s.eye(2)+laplacian
    delta = s.Matrix(s.symbols("d0:2",real=True))
    origin = s.Matrix(s.symbols("o0:2",real=True))
    center = s.Matrix(s.symbols("c0:2",real=True))
    shifted = ((origin+delta-center).T*precision*(origin+delta-center))[0]/2
    assert s.hessian(shifted,tuple(delta)) == precision
    assert s.expand(((delta.T*laplacian*delta)[0])-(delta[0]-delta[1])**2) == 0
    assert s.factor(precision.det()) == ridge*(ridge+2)
    checks.append("full-coordinate delta reparameterization preserves quadratic curvature; graph PSD plus ridge>0 is SPD in fixture")

    a,b = s.symbols("a b",positive=True)
    x,y = s.symbols("x y",real=True)
    vector=s.Matrix([x,y])
    for sign_a,sign_b in itertools.product((-1,1),repeat=2):
        row=s.Matrix([sign_a*a,sign_b*b])
        majorizer=s.diag(a*(a+b),b*(a+b))
        remainder=(vector.T*(majorizer-row*row.T)*vector)[0]
        assert s.expand(remainder-a*b*(x-sign_a*sign_b*y)**2)==0
    checks.append("absolute-row Cauchy majorizer verified for all two-coordinate sign patterns; zero and general-dimensional cases follow from weighted Cauchy")

    shift=s.symbols("bias_shift",real=True)
    pos,neg=s.symbols("positive negative",real=True)
    assert s.expand((pos+shift)-(neg+shift)-(pos-neg))==0
    checks.append("common item-bias shifts are invisible to pair margins; a unique regularized optimum is not data identification")

    # S=1, two equiprobable candidate exponentials 1 and 3; corrected partition 2 or 6.
    sampled=(s.log(3)+s.log(7))/2
    full=s.log(5)
    partitions = s.Matrix([2, 6])
    assert sum(partitions)/2 == 4 and s.Integer(21)<25
    assert s.simplify(s.exp(2*sampled))==21 and s.exp(2*full)==25
    checks.append("sample partition sum is unbiased but expected log-loss is smaller than complete-list loss in an exact two-point counterexample")

    p,w,z=s.symbols("p w z",positive=True)
    risk=w*p*(1-z)**2+(1-p)*z*z
    optimum=w*p/(1+(w-1)*p)
    assert s.factor(s.diff(risk,z).subs(z,optimum))==0
    assert s.factor(s.diff(optimum,p)-w/(1+(w-1)*p)**2)==0
    checks.append("weighted square population optimum is monotone for 0<p<1,w>0; objective mismatch is a hypothesis, not a universal flaw")
    return {"status":"pass","sympy":s.__version__,"groups":len(checks),"checks":checks,
            "new_training":False,"dataset_read":False,
            "scope":"exact identities and finite fixtures; no generalization, NDCG improvement or novelty proof"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=check()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
