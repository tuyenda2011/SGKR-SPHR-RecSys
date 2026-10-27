"""Exact checks for proposed-model claims; not a ranking-performance proof."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.tmp' / 'sympy114'))

import sympy as sp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sp.__version__ == '1.14.0'
    alpha, mg, mk = sp.symbols('alpha m_g m_k', real=True)
    regularization = sp.symbols('lambda', positive=True)
    margin = (1-alpha)*mg + alpha*mk
    loss = sp.log(1+sp.exp(-margin)) + regularization*alpha**2/2
    first = regularization*alpha - (mk-mg)/(1+sp.exp(margin))
    second = regularization + (mk-mg)**2*sp.exp(margin)/(1+sp.exp(margin))**2
    assert sp.simplify(sp.diff(loss, alpha)-first) == 0
    assert sp.simplify(sp.diff(loss, alpha, 2)-second) == 0
    assert second.is_positive is True
    assert sp.simplify(loss.subs(alpha, 0)-sp.log(1+sp.exp(-mg))) == 0
    r0, r1, r2 = sp.symbols('r_0 r_1 r_2', real=True)
    entries = sp.symbols('v:6', real=True)
    factors = sp.Matrix(3, 2, entries)
    gram = factors*factors.T
    weights = gram - sp.diag(*gram.diagonal())
    scores = sp.Matrix([[r0, r1, r2]])*weights
    assert all(weights[index, index] == 0 for index in range(3))
    assert all(sp.diff(scores[index], value) == 0 for index, value in enumerate((r0, r1, r2)))
    before, after = sp.Rational(3, 10), sp.Rational(4, 5)
    assert sp.exp(-after) < sp.exp(-before)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'sympy_version': sp.__version__, 'status': 'pass',
              'fixed_score_mixture': {'margin': str(margin), 'loss': str(loss), 'first_derivative': str(first),
                                      'second_derivative': str(second), 'strictly_convex_when_lambda_positive': True,
                                      'zero_coefficient_recovers_baseline': True,
                                      'limits': 'Fixed branch scores; nonlinear learned gates need not be convex. Lower train loss does not guarantee ranking gain.'},
              'diagonal_zero': {'matrix': str(weights), 'own_input_partial_derivative': 0,
                                'limits': 'At fixed item weights only. Fitting weights on all train edges can still encode targets indirectly; not a leakage proof.'},
              'training_vs_ranking_counterexample': {'items': ['train_positive_masked_at_validation', 'validation_target', 'irrelevant'],
                                                     'before_scores': ['1/2', '3/10', '1/5'], 'after_scores': ['1', '1/10', '1/5'],
                                                     'train_positive_minus_irrelevant_margin_before': str(before),
                                                     'train_positive_minus_irrelevant_margin_after': str(after),
                                                     'train_bpr_loss_decreases': True, 'validation_NDCG_at_1_before': 1,
                                                     'validation_NDCG_at_1_after': 0,
                                                     'limits': 'Constructed counterexample; not an estimate from Amazon data.'}}
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
