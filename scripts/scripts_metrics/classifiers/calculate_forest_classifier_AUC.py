'''
Random forest on the embeddings of one model, with stratified k-fold cross-validation.
'''
import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold

from bend.utils.metrics_utils import (add_single_model_args, single_model_paths, load_pairs, load_annotation,
                                      build_features, mean_std, print_summary, results_dir, save_results,
                                      summary_fields)


def cross_validate(X, y, args):
    '''AUROC, AUPRC and validation loss of each fold.'''
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                     random_state=args.seed, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        probs = clf.predict_proba(X[te])[:, 1]

        rows.append({'fold': fold, 'AUC': roc_auc_score(y[te], probs),
                     'AUPRC': average_precision_score(y[te], probs), 'val_loss': log_loss(y[te], probs, labels=[0, 1])})
        print(f"Fold {fold}: AUC = {rows[-1]['AUC']:.4f} | AUPRC = {rows[-1]['AUPRC']:.4f} | val loss = {rows[-1]['val_loss']:.4f}")
    return rows


def main():

    parser = argparse.ArgumentParser('Cross-validated random forest on the variant embeddings of one model')
    add_single_model_args(parser, default_features=['diff'])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--max_depth', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    emb_path, labels_path = single_model_paths(parser, args)

    idx, ref, alt = load_pairs(emb_path)
    y = load_annotation(labels_path, idx)['label'].to_numpy()
    print(f'Embeddings: {emb_path}')
    print(f'Labels:     {labels_path}')
    print(f'{len(y)} variants ({y.sum()} positive) | embedding size {ref.shape[1]}')

    summary, all_rows, log_fields = {}, [], []
    for feature in args.features:
        X = build_features(ref, alt, feature)
        print(f'\n=== features: {feature} | X: {X.shape}')
        rows = cross_validate(X, y, args)

        summary[feature] = {'AUC': mean_std([r['AUC'] for r in rows]), 'AUPRC': mean_std([r['AUPRC'] for r in rows])}
        print(f"Mean AUC: {summary[feature]['AUC']} | Mean AUPRC: {summary[feature]['AUPRC']}")
        all_rows += [{'features': feature, **r} for r in rows]
        log_fields.append({'out_file': args.out_file, 'pooling': args.pooling, 'features': feature,
                           'embeddings': emb_path, **summary_fields(rows, y), 'folds': args.folds,
                           'n_estimators': args.n_estimators, 'max_depth': args.max_depth, 'seed': args.seed})

    print_summary(f'Random forest | {args.model} | {args.dataset} | {args.out_file} | {args.pooling}'
                  f' (AUPRC of a random classifier: {y.mean():.4f})', summary)
    save_results(results_dir(__file__, args.model, args.dataset), f'{args.out_file}_{args.pooling}',
                 pd.DataFrame(all_rows), log_fields)


if __name__ == '__main__':
    main()
