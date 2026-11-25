'''
Ensemble 1: raw and z-scored mean of the distances of several models (no training).
'''
import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from bend.utils.metrics_utils import (add_ensemble_args, score_name, load_ensemble_scores, print_summary, results_dir,
                                      save_results, fold_std)


def main():

    parser = argparse.ArgumentParser('Ensemble: average of the distances of several models')
    add_ensemble_args(parser, default_models=['dnabert2', 'grover', 'hyenadna', 'ag'])
    parser.add_argument('--score', type=score_name, default='cosine_distance', help='Distance of each model: cosine_distance, avg_n{n}, ccs_n{n} or cls')
    parser.add_argument('--folds', type=int, default=5, help='Folds for the +/- (std over the same folds as the classifiers)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    scores, y, out_files = load_ensemble_scores(parser, args)
    print(f'{len(y)} variants ({y.sum()} positive)')

    z_scores = (scores - scores.mean()) / scores.std(ddof=0)
    combined = {model: scores[model] for model in scores.columns}  # each model alone, for reference
    combined['raw mean'] = scores.mean(axis=1)
    combined['z-scored mean'] = z_scores.mean(axis=1)

    rows = [{'score_of': name, 'AUC': roc_auc_score(y, s), 'AUPRC': average_precision_score(y, s),
             **dict(zip(('AUC_std', 'AUPRC_std'), fold_std(y, s.to_numpy(), args.folds, args.seed)))}
            for name, s in combined.items()]
    print_summary(f'Average of distances | {args.dataset} | {args.score}'
                  f' (AUPRC of a random classifier: {y.mean():.4f})',
                  {r['score_of']: {'AUC': f"{r['AUC']:.4f} +/- {r['AUC_std']:.4f}",
                                   'AUPRC': f"{r['AUPRC']:.4f} +/- {r['AUPRC_std']:.4f}"} for r in rows})

    by_name = {r['score_of']: r for r in rows}
    log = {'out_files': '+'.join(out_files), 'score': args.score, 'n_variants': len(y), 'chance_AUPRC': f'{y.mean():.4f}',
           **{f'{metric}_{key}': f"{by_name[name][metric]:.4f}+/-{by_name[name][metric + '_std']:.4f}"
              for key, name in (('raw_mean', 'raw mean'), ('z_mean', 'z-scored mean')) for metric in ('AUC', 'AUPRC')},
           'folds': args.folds, 'seed': args.seed}
    save_results(results_dir(__file__, args.models, args.dataset), args.score, pd.DataFrame(rows), [log])


if __name__ == '__main__':
    main()
