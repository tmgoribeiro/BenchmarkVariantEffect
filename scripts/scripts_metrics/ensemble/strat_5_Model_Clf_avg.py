'''
Ensemble 5: one MLP per model and the mean of their out-of-fold predictions.
'''
import argparse

import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score

from bend.utils.metrics_utils import (add_ensemble_args, add_embedding_args, add_mlp_args, load_ensemble_embeddings,
                                      build_features, seed_everything, get_device, out_of_fold_logits, print_summary,
                                      results_dir, save_results, mlp_settings, fold_std)


def main():

    parser = argparse.ArgumentParser('Ensemble: one MLP per model, average of their predictions')
    add_ensemble_args(parser, default_models=['dnabert2', 'grover', 'hyenadna', 'ag'])
    add_embedding_args(parser, default_features=['diff'])
    add_mlp_args(parser)
    args = parser.parse_args()

    device = get_device()
    pairs, y, out_files = load_ensemble_embeddings(parser, args)
    print(f'{len(y)} variants ({y.sum()} positive) | device: {device}')

    all_rows, log_fields = [], []
    for feature in args.features:
        seed_everything(args.seed)
        feats = {m: build_features(ref, alt, feature) for m, (ref, alt) in pairs.items()}
        print(f'\n=== features: {feature}')
        logits = out_of_fold_logits(feats, y, args, device)

        # each model alone (out-of-fold over all variants), for reference, then the two averages
        combined = {model: logits[:, m] for m, model in enumerate(feats)}
        combined['mean of probabilities'] = expit(logits).mean(axis=1)
        combined['mean of logits'] = logits.mean(axis=1)

        # out-of-fold predictions over all variants; the +/- is the std over the same folds
        rows = [{'features': feature, 'prediction_of': name, 'AUC': roc_auc_score(y, s),
                 'AUPRC': average_precision_score(y, s),
                 **dict(zip(('AUC_std', 'AUPRC_std'), fold_std(y, s, args.folds, args.seed)))}
                for name, s in combined.items()]
        print_summary(f'Average of per-model MLPs | {args.dataset} | {args.pooling} | {feature}'
                      f' (AUPRC of a random classifier: {y.mean():.4f})',
                      {r['prediction_of']: {'AUC': f"{r['AUC']:.4f} +/- {r['AUC_std']:.4f}",
                                            'AUPRC': f"{r['AUPRC']:.4f} +/- {r['AUPRC_std']:.4f}"} for r in rows})

        all_rows += rows
        by_name = {r['prediction_of']: r for r in rows}
        log_fields.append({
            'out_files': '+'.join(out_files), 'pooling': args.pooling, 'features': feature, 'n_variants': len(y),
            'chance_AUPRC': f'{y.mean():.4f}',
            **{f'{metric}_{key}': f"{by_name[name][metric]:.4f}+/-{by_name[name][metric + '_std']:.4f}"
               for key, name in (('mean_probs', 'mean of probabilities'), ('mean_logits', 'mean of logits'))
               for metric in ('AUC', 'AUPRC')},
            **mlp_settings(args), 'device': device,
        })

    save_results(results_dir(__file__, args.models, args.dataset), args.pooling, pd.DataFrame(all_rows), log_fields)


if __name__ == '__main__':
    main()
