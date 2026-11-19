'''
AUROC and AUPRC of the cosine distances, over all variants and per consequence class.

Reads the csv files written by predict_variant_effects.py (cosine_distance_<out_file>.csv) or by
predict_variant_effects_pooling_strategies.py (pooling_strategies/<strategy>_<out_file>.csv). Every
column whose name starts with "distance" is scored, unless --columns is given.

    python scripts/scripts_metrics/calculate_cosine_metrics.py runs/dnabert2/disease_dnabert2/cosine_distance_disease_dnabert2.csv
'''
import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from count_consequences import read_table, find_consequence_column


def scores(labels: pd.Series, distances: pd.Series) -> dict:
    '''AUROC and AUPRC, or NaN when there is only one class (both metrics need positives and negatives).'''
    distances = distances.astype(float)
    if labels.nunique() < 2:
        return {'AUC': float('nan'), 'AUPRC': float('nan')}
    return {'AUC': roc_auc_score(labels, distances), 'AUPRC': average_precision_score(labels, distances)}


def scores_table(df: pd.DataFrame, score_column: str, consequence_column: str) -> pd.DataFrame:
    '''One row per consequence class plus an "all" row with the overall scores.'''
    rows = {'all': {'n': len(df), 'positives': int(df['label'].sum()), **scores(df['label'], df[score_column])}}
    if consequence_column is not None:
        for consequence, sub_df in df.groupby(consequence_column):
            rows[consequence] = {'n': len(sub_df), 'positives': int(sub_df['label'].sum()),
                                 **scores(sub_df['label'], sub_df[score_column])}
    table = pd.DataFrame.from_dict(rows, orient='index')
    # overall row first, then the classes from the largest to the smallest
    return pd.concat([table.loc[['all']], table.drop(index='all').sort_values('n', ascending=False)])


def main():

    parser = argparse.ArgumentParser('AUROC and AUPRC per consequence')
    parser.add_argument('files', type=str, nargs='+', help='csv files written by the prediction scripts')
    parser.add_argument('--columns', type=str, nargs='+', default=None, help='Score columns. Defaults to every column starting with "distance"')
    parser.add_argument('--consequence_column', type=str, default=None, help='Name of the consequence column. Defaults to the column called consequence (any case)')

    args = parser.parse_args()

    for path in args.files:
        df = read_table(path)
        score_columns = args.columns if args.columns is not None else [c for c in df.columns if c.startswith('distance')]
        consequence_column = find_consequence_column(df, args.consequence_column)

        print(f'\n=== {path}')
        if consequence_column is None:
            print('No consequence column, only the overall scores are shown')

        for score_column in score_columns:
            table = scores_table(df, score_column, consequence_column)
            print(f'\n{score_column}  (classes with only positives or only negatives have no score)')
            print(table.to_string(float_format='{:.4f}'.format))


if __name__ == '__main__':
    main()
