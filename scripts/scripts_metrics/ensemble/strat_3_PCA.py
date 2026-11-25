'''
Ensemble 3: the embeddings of all models, reduced together with one PCA, then an MLP (k-fold).
'''
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from bend.utils.metrics_utils import (add_ensemble_args, add_embedding_args, add_mlp_args, load_ensemble_embeddings,
                                      build_features, seed_everything, get_device, to_tensor, train_mlp,
                                      predict_logits, bce, mean_std, print_summary, now, peak_ram_gb, peak_gpu_gb,
                                      results_dir, save_results, mlp_settings, summary_fields)


def joint_pca(feats, tr, te, n_components, device):
    '''Per-model scaling, concatenation, centring and PCA, all fitted on the training variants only.'''
    tr_blocks, te_blocks = [], []
    for Xm in feats.values():
        scaler = StandardScaler().fit(Xm[tr])
        w = np.sqrt(Xm.shape[1])  # equalise each model's total variance
        tr_blocks.append(scaler.transform(Xm[tr]) / w)
        te_blocks.append(scaler.transform(Xm[te]) / w)

    Xc_tr = to_tensor(np.concatenate(tr_blocks, axis=1), device)
    Xc_te = to_tensor(np.concatenate(te_blocks, axis=1), device)
    mean = Xc_tr.mean(dim=0, keepdim=True)
    Xc_tr -= mean
    Xc_te -= mean

    q = min(n_components, *Xc_tr.shape)
    _, _, V = torch.pca_lowrank(Xc_tr, q=q, center=False, niter=4)  # V: (D, q)
    X_tr, X_te = Xc_tr @ V, Xc_te @ V

    del Xc_tr, Xc_te, V
    if device == 'cuda':
        torch.cuda.empty_cache()
    return X_tr, X_te


def run(feats, y, args, device):
    '''Metrics and timings of each fold.'''
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        fold_start = now()
        X_tr, X_te = joint_pca(feats, tr, te, args.n_components, device)
        pca_time = now() - fold_start
        print(f'Fold {fold} | X: {tuple(X_tr.shape)}')

        y_tr, y_te = to_tensor(y[tr], device), to_tensor(y[te], device)
        train_start = now()
        model = train_mlp(X_tr, y_tr, args, desc=f'Fold {fold}')
        train_time = now() - train_start

        predict_start = now()
        logits = predict_logits(model, X_te)
        probs = torch.sigmoid(logits).cpu().numpy()
        predict_time = now() - predict_start

        rows.append({'fold': fold, 'X_dim': X_tr.shape[1], 'AUC': roc_auc_score(y[te], probs),
                     'AUPRC': average_precision_score(y[te], probs), 'val_loss': bce(logits, y_te),
                     'pca_s': pca_time, 'train_s': train_time, 'predict_s': predict_time})
        r = rows[-1]
        print(f"Fold {fold}: AUC = {r['AUC']:.4f} | AUPRC = {r['AUPRC']:.4f} | val loss = {r['val_loss']:.4f} | "
              f"pca {pca_time:.1f}s | train {train_time:.1f}s | predict {predict_time:.3f}s\n")
    return rows


def main():

    parser = argparse.ArgumentParser('Ensemble: joint PCA of the embeddings of several models + MLP')
    add_ensemble_args(parser, default_models=['dnabert2', 'grover', 'hyenadna', 'ag'])
    add_embedding_args(parser, default_features=['diff'])
    parser.add_argument('--n_components', type=int, default=512, help='PCA dimensions kept for the concatenation of all models')
    add_mlp_args(parser)
    args = parser.parse_args()

    device = get_device()
    start = now()
    pairs, y, out_files = load_ensemble_embeddings(parser, args)
    load_time = now() - start
    print(f'{len(y)} variants ({y.sum()} positive) | device: {device}')

    summary, all_rows, log_fields = {}, [], []
    for feature in args.features:
        seed_everything(args.seed)
        feats = {m: build_features(ref, alt, feature) for m, (ref, alt) in pairs.items()}
        print(f'\n=== features: {feature} | ' + ', '.join(f'{m} {X.shape[1]}' for m, X in feats.items()))

        feature_start = now()
        rows = run(feats, y, args, device)
        elapsed = now() - feature_start
        x_dim = rows[-1]['X_dim']
        pca_t, train_t, predict_t = (np.mean([r[k] for r in rows]) for k in ('pca_s', 'train_s', 'predict_s'))
        summary[feature] = {'AUC': mean_std([r['AUC'] for r in rows]), 'AUPRC': mean_std([r['AUPRC'] for r in rows]), 'X_dim': x_dim}

        print(f"Mean AUC: {summary[feature]['AUC']} | Mean AUPRC: {summary[feature]['AUPRC']}")
        print(f'Time: total {elapsed:.1f}s | load {load_time:.1f}s | per fold: pca {pca_t:.1f}s, '
              f'train {train_t:.1f}s, predict {predict_t:.3f}s')
        print(f'Peak memory: GPU {peak_gpu_gb(device):.2f} GB | RAM {peak_ram_gb():.2f} GB | device: {device}')

        all_rows += [{'features': feature, **r} for r in rows]
        log_fields.append({
            'out_files': '+'.join(out_files), 'pooling': args.pooling, 'features': feature,
            'n_components': args.n_components, 'X_dim': x_dim, **summary_fields(rows, y), **mlp_settings(args),
            'elapsed': f'{elapsed:.1f}s', 'load': f'{load_time:.1f}s', 'pca_per_fold': f'{pca_t:.1f}s',
            'train_per_fold': f'{train_t:.1f}s', 'predict_per_fold': f'{predict_t:.3f}s',
            'peak_gpu': f'{peak_gpu_gb(device):.2f}GB', 'peak_ram': f'{peak_ram_gb():.2f}GB', 'device': device,
        })

    print_summary(f'Joint PCA + MLP | {args.dataset} | {"+".join(args.models)} | {args.pooling}'
                  f' (AUPRC of a random classifier: {y.mean():.4f})', summary)
    save_results(results_dir(__file__, args.models, args.dataset), f'{args.pooling}_pca{args.n_components}',
                 pd.DataFrame(all_rows), log_fields)


if __name__ == '__main__':
    main()
