'''
Spearman and Kendall correlations between each distance column and the measured effect |logFC|.
'''
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr, kendalltau

from bend.utils.metrics_utils import read_table, resolve_run, results_dir, save_results


def main():

    parser = argparse.ArgumentParser('Correlation between |logFC| and the distances')
    parser.add_argument('files', type=str, nargs='+', help='csv files written by the prediction scripts')
    parser.add_argument('--columns', type=str, nargs='+', default=None, help='Score columns. Defaults to every column starting with "distance"')
    parser.add_argument('--effect_column', type=str, default='logFC', help='Column with the measured effect, compared in absolute value. Default logFC')
    parser.add_argument('--model', type=str, default=None, help='Model folder of the results. Defaults to the model in the path of each csv')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset folder of the results. Defaults to the dataset in the path of each csv')

    args = parser.parse_args()

    for path in args.files:
        model, dataset = resolve_run(parser, path, args.model, args.dataset)
        df = read_table(path)
        if args.effect_column not in df.columns:
            parser.error(f'No column {args.effect_column} in {path}')
        score_columns = args.columns if args.columns is not None else [c for c in df.columns if c.startswith('distance')]

        print(f'\n=== {path}')
        rows, log = [], {'file': path, 'effect_column': args.effect_column}
        for column in score_columns:
            sub = df[[args.effect_column, column]].astype(float).dropna()
            effect = sub[args.effect_column].abs()
            spearman, kendall = spearmanr(effect, sub[column]), kendalltau(effect, sub[column])
            rows.append({'score': column, 'n': len(sub),
                         'spearman': spearman.statistic, 'spearman_p': spearman.pvalue,
                         'kendall': kendall.statistic, 'kendall_p': kendall.pvalue})
            print(f'{column}: Spearman {spearman.statistic:.4f} (p = {spearman.pvalue:.3g}) | '
                  f'Kendall {kendall.statistic:.4f} (p = {kendall.pvalue:.3g}) | n = {len(sub)}')
            log[f'spearman_{column}'] = f'{spearman.statistic:.4f}'
            log[f'kendall_{column}'] = f'{kendall.statistic:.4f}'

        save_results(results_dir(__file__, model, dataset), Path(path).stem, pd.DataFrame(rows), [log])


if __name__ == '__main__':
    main()
