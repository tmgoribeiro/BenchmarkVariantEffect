'''
MLP (or linear classifier) on the embeddings of one model, with stratified k-fold cross-validation.
'''
import argparse

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from bend.utils.metrics_utils import (add_single_model_args, add_mlp_args, single_model_paths, load_pairs,
                                      load_annotation, build_features, seed_everything, get_device, to_tensor,
                                      train_mlp, predict_logits, bce, mean_std, print_summary, results_dir,
                                      save_results, mlp_settings, summary_fields)


def cross_validate(X, y, args, device):
    '''AUROC, AUPRC and validation loss of each fold.'''
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        # scale on train only (no leakage)
        scaler = StandardScaler().fit(X[tr])
        X_tr, X_te = to_tensor(scaler.transform(X[tr]), device), to_tensor(scaler.transform(X[te]), device)
        y_tr, y_te = to_tensor(y[tr], device), to_tensor(y[te], device)

        model = train_mlp(X_tr, y_tr, args, desc=f'Fold {fold}')
        logits = predict_logits(model, X_te)
        probs = torch.sigmoid(logits).cpu().numpy()

        rows.append({'fold': fold, 'AUC': roc_auc_score(y[te], probs),
                     'AUPRC': average_precision_score(y[te], probs), 'val_loss': bce(logits, y_te)})
        print(f"Fold {fold}: AUC = {rows[-1]['AUC']:.4f} | AUPRC = {rows[-1]['AUPRC']:.4f} | val loss = {rows[-1]['val_loss']:.4f}\n")
    return rows


def main():

    parser = argparse.ArgumentParser('Cross-validated MLP on the variant embeddings of one model')
    add_single_model_args(parser, default_features=['diff'])
    add_mlp_args(parser)
    args = parser.parse_args()

    emb_path, labels_path = single_model_paths(parser, args)
    device = get_device()

    idx, ref, alt = load_pairs(emb_path)
    y = load_annotation(labels_path, idx)['label'].to_numpy()
    print(f'Embeddings: {emb_path}')
    print(f'Labels:     {labels_path}')
    print(f'{len(y)} variants ({y.sum()} positive) | embedding size {ref.shape[1]} | device: {device}')

    summary, all_rows, log_fields = {}, [], []
    for feature in args.features:
        seed_everything(args.seed)
        X = build_features(ref, alt, feature)
        print(f'\n=== features: {feature} | X: {X.shape}')
        rows = cross_validate(X, y, args, device)

        summary[feature] = {'AUC': mean_std([r['AUC'] for r in rows]), 'AUPRC': mean_std([r['AUPRC'] for r in rows])}
        print(f"Mean AUC: {summary[feature]['AUC']} | Mean AUPRC: {summary[feature]['AUPRC']}")
        all_rows += [{'features': feature, **r} for r in rows]
        log_fields.append({'out_file': args.out_file, 'pooling': args.pooling, 'features': feature,
                           'embeddings': emb_path, **summary_fields(rows, y), **mlp_settings(args), 'device': device})

    print_summary(f'MLP | {args.model} | {args.dataset} | {args.out_file} | {args.pooling}'
                  f' (AUPRC of a random classifier: {y.mean():.4f})', summary)
    save_results(results_dir(__file__, args.model, args.dataset), f'{args.out_file}_{args.pooling}',
                 pd.DataFrame(all_rows), log_fields)


if __name__ == '__main__':
    main()
