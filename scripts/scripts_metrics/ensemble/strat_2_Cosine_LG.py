'''
Ensemble 2: logistic regression on the distances of several models, with k-fold cross-validation.
'''
import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bend.utils.metrics_utils import (add_ensemble_args, score_name, load_ensemble_scores, mean_std, results_dir,
                                      save_results, summary_fields)


def main():

    parser = argparse.ArgumentParser('Ensemble: logistic regression on the distances of several models')
    add_ensemble_args(parser, default_models=['dnabert2', 'grover', 'hyenadna', 'ag'])
    parser.add_argument('--score', type=score_name, default='cosine_distance', help='Distance of each model: cosine_distance, avg_n{n}, ccs_n{n} or cls')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--C', type=float, default=1.0, help='Inverse regularisation strength of the logistic regression')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    scores, y, out_files = load_ensemble_scores(parser, args)
    X = scores.to_numpy()
    print(f'{len(y)} variants ({y.sum()} positive) | weights in the order {list(scores.columns)}\n')

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        stacker = make_pipeline(StandardScaler(), LogisticRegression(C=args.C)).fit(X[tr], y[tr])
        s = stacker.decision_function(X[te])  # the stacker's logit
        weights = dict(zip(scores.columns, stacker[-1].coef_[0]))
        rows.append({'fold': fold, 'AUC': roc_auc_score(y[te], s), 'AUPRC': average_precision_score(y[te], s),
                     **{f'weight_{m}': w for m, w in weights.items()}, 'bias': stacker[-1].intercept_[0]})
        print(f"Fold {fold}: AUC = {rows[-1]['AUC']:.4f} | AUPRC = {rows[-1]['AUPRC']:.4f} | "
              f"weights: {stacker[-1].coef_[0].round(3)} | bias: {stacker[-1].intercept_.round(3)}")

    print(f'\n=== Logistic regression on distances | {args.dataset} | {args.score}')
    print(f"AUC:   {mean_std([r['AUC'] for r in rows])}")
    print(f"AUPRC: {mean_std([r['AUPRC'] for r in rows])} (AUPRC of a random classifier: {y.mean():.4f})")

    log = {'out_files': '+'.join(out_files), 'score': args.score, **summary_fields(rows, y),
           'folds': args.folds, 'C': args.C, 'seed': args.seed}
    save_results(results_dir(__file__, args.models, args.dataset), args.score, pd.DataFrame(rows), [log])


if __name__ == '__main__':
    main()
