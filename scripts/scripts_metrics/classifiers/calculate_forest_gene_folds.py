'''
Random forest on the embeddings of one model, tested on held-out groups of chromosomes.
Uses cuML on the GPU when it is installed.
'''
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

try:
    from cuml.ensemble import RandomForestClassifier
    BACKEND = 'cuml'
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    BACKEND = 'sklearn'

from bend.utils.metrics_utils import (add_single_model_args, single_model_paths, load_pairs, load_annotation,
                                      build_features, mean_std, print_summary, results_dir, save_results,
                                      summary_fields)


# chromosomes held out together in each fold (without the chr prefix)
TEST_GROUPS = [
    {'3', '6', '9', '12', '16', '18', '19', '21'},
    {'2', '5', '11', '14', '17', '20', '22', 'X'},
    {'1', '4', '7', '8', '10', '13', '15'},
]


def chromosome_holdout(X, y, chrom, args):
    '''AUROC, AUPRC, chance AUPRC and validation loss of each chromosome group.'''
    rows = []
    for g, test_set in enumerate(TEST_GROUPS, 1):
        in_test = np.isin(chrom, list(test_set))
        te, tr = np.where(in_test)[0], np.where(~in_test)[0]
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            print(f'Group {g}: skipped, its test or training variants do not have both labels (n_test = {len(te)})')
            continue

        kwargs = {} if BACKEND == 'cuml' else {'n_jobs': -1}
        clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                     random_state=args.seed, **kwargs)
        clf.fit(X[tr], y[tr])
        probs = clf.predict_proba(X[te])
        probs = probs.get() if hasattr(probs, 'get') else np.asarray(probs)  # cupy -> numpy
        probs = probs[:, 1]

        rows.append({'group': g, 'chromosomes': ','.join(sorted(test_set, key=lambda c: (not c.isdigit(), c.zfill(2)))),
                     'n_test': len(te), 'AUC': roc_auc_score(y[te], probs),
                     'AUPRC': average_precision_score(y[te], probs), 'chance_AUPRC': y[te].mean(),
                     'val_loss': log_loss(y[te], probs, labels=[0, 1])})
        r = rows[-1]
        print(f"Group {g}: AUC = {r['AUC']:.4f} | AUPRC = {r['AUPRC']:.4f} | baseline = {r['chance_AUPRC']:.4f} | "
              f"val loss = {r['val_loss']:.4f} | n_test = {r['n_test']}")
    return rows


def main():

    parser = argparse.ArgumentParser('Random forest on the variant embeddings of one model, with chromosome holdout')
    add_single_model_args(parser, default_features=['avg'])
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--max_depth', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    emb_path, labels_path = single_model_paths(parser, args)

    idx, ref, alt = load_pairs(emb_path)
    annotation = load_annotation(labels_path, idx)
    y = annotation['label'].to_numpy()
    chrom = annotation['chromosome'].astype(str).str.replace('chr', '').to_numpy()
    print(f'Embeddings: {emb_path}')
    print(f'Labels:     {labels_path}')
    print(f'{len(y)} variants ({y.sum()} positive) | embedding size {ref.shape[1]} | random forest: {BACKEND}')

    never_tested = ~np.isin(chrom, list(set().union(*TEST_GROUPS)))
    if never_tested.any():
        print(f'{never_tested.sum()} variants on chromosomes outside every test group '
              f'({sorted(set(chrom[never_tested]))}), they are only used for training')

    summary, all_rows, log_fields = {}, [], []
    for feature in args.features:
        X = build_features(ref, alt, feature)
        print(f'\n=== features: {feature} | X: {X.shape}')
        rows = chromosome_holdout(X, y, chrom, args)
        if not rows:
            parser.error('No chromosome group has both labels among its variants, nothing to evaluate')

        summary[feature] = {'AUC': mean_std([r['AUC'] for r in rows]), 'AUPRC': mean_std([r['AUPRC'] for r in rows])}
        print(f"Mean AUC: {summary[feature]['AUC']} | Mean AUPRC: {summary[feature]['AUPRC']}")
        all_rows += [{'features': feature, **r} for r in rows]
        log_fields.append({'out_file': args.out_file, 'pooling': args.pooling, 'features': feature,
                           'embeddings': emb_path, **summary_fields(rows, y), 'n_estimators': args.n_estimators,
                           'max_depth': args.max_depth, 'seed': args.seed, 'backend': BACKEND})

    print_summary(f'Random forest, chromosome holdout | {args.model} | {args.dataset} | {args.out_file} | {args.pooling}'
                  f' (AUPRC of a random classifier: {y.mean():.4f})', summary)
    save_results(results_dir(__file__, args.model, args.dataset), f'{args.out_file}_{args.pooling}',
                 pd.DataFrame(all_rows), log_fields)


if __name__ == '__main__':
    main()
