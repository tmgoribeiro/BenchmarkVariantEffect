'''
Ensemble 6: one MLP per model and a logistic regression on their out-of-fold logits (stacking).
'''
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

from bend.utils.metrics_utils import (add_ensemble_args, add_embedding_args, add_mlp_args, load_ensemble_embeddings,
                                      build_features, seed_everything, get_device, out_of_fold_logits, mean_std,
                                      print_summary, results_dir, save_results, mlp_settings, summary_fields)


def main():

    parser = argparse.ArgumentParser('Ensemble: one MLP per model, logistic regression on their logits')
    add_ensemble_args(parser, default_models=['dnabert2', 'grover', 'hyenadna', 'ag'])
    add_embedding_args(parser, default_features=['diff'])
    add_mlp_args(parser)
    parser.add_argument('--C', type=float, default=1.0, help='Inverse regularisation strength of the logistic regression')
    args = parser.parse_args()

    device = get_device()
    pairs, y, out_files = load_ensemble_embeddings(parser, args)
    print(f'{len(y)} variants ({y.sum()} positive) | device: {device}')

    summary, all_rows, log_fields = {}, [], []
    for feature in args.features:
        seed_everything(args.seed)
        feats = {m: build_features(ref, alt, feature) for m, (ref, alt) in pairs.items()}
        print(f'\n=== features: {feature}')
        logits = out_of_fold_logits(feats, y, args, device)

        print(f'\nStacker, weights in the order {list(feats)}')
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        rows = []
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
            stacker = LogisticRegression(C=args.C).fit(logits[tr], y[tr])
            s = stacker.decision_function(logits[te])  # the stacker's logit
            rows.append({'fold': fold, 'AUC': roc_auc_score(y[te], s), 'AUPRC': average_precision_score(y[te], s),
                         **{f'weight_{m}': w for m, w in zip(feats, stacker.coef_[0])}, 'bias': stacker.intercept_[0]})
            print(f"Fold {fold}: AUC = {rows[-1]['AUC']:.4f} | AUPRC = {rows[-1]['AUPRC']:.4f} | "
                  f"weights: {stacker.coef_[0].round(3)} | bias: {stacker.intercept_.round(3)}")

        summary[feature] = {'AUC': mean_std([r['AUC'] for r in rows]), 'AUPRC': mean_std([r['AUPRC'] for r in rows])}
        all_rows += [{'features': feature, **r} for r in rows]
        log_fields.append({'out_files': '+'.join(out_files), 'pooling': args.pooling, 'features': feature,
                           **summary_fields(rows, y), **mlp_settings(args), 'C': args.C, 'device': device})

    print_summary(f'Stacking of per-model MLPs | {args.dataset} | {"+".join(args.models)} | {args.pooling}'
                  f' (AUPRC of a random classifier: {y.mean():.4f})', summary)
    save_results(results_dir(__file__, args.models, args.dataset), args.pooling, pd.DataFrame(all_rows), log_fields)


if __name__ == '__main__':
    main()
