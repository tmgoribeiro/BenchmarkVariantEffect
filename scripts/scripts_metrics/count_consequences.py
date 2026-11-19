'''
List the consequence classes of one or more variant datasets, with the number of variants and of
positives (label = 1) in each class.

    python scripts/scripts_metrics/count_consequences.py data/variant_effects/variant_effects_disease.bed data/variant_effects/variant_effects_expression.bed
'''
import argparse
from pathlib import Path

import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    '''Bed files are tab-separated; the csv files of the prediction scripts are comma-separated with the index first.'''
    if Path(path).suffix == '.csv':
        return pd.read_csv(path, index_col=0)
    return pd.read_csv(path, sep='\t')


def find_consequence_column(df: pd.DataFrame, column: str = None) -> str:
    '''Name of the consequence column, or None if there is none.'''
    if column is not None:
        return column if column in df.columns else None
    return next((c for c in df.columns if c.lower() == 'consequence'), None)


def consequence_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    '''One row per consequence: number of variants, number of positives and their fraction.'''
    counts = df.groupby(column)['label'].agg(n='size', positives='sum')
    counts['positive_fraction'] = (counts['positives'] / counts['n']).round(3)
    return counts.sort_values('n', ascending=False)


def main():

    parser = argparse.ArgumentParser('List the consequence classes of variant datasets')
    parser.add_argument('files', type=str, nargs='+', help='Bed or csv files of the datasets')
    parser.add_argument('--column', type=str, default=None, help='Name of the consequence column. Defaults to the column called consequence (any case)')

    args = parser.parse_args()

    for path in args.files:
        df = read_table(path)
        column = find_consequence_column(df, args.column)
        if column is None:
            print(f'\n{path}: no consequence column. Columns: {list(df.columns)}')
            continue

        counts = consequence_counts(df, column)
        print(f'\n{path}: {len(df)} variants, {int(df["label"].sum())} positive, {len(counts)} consequence classes')
        print(counts.to_string())


if __name__ == '__main__':
    main()
