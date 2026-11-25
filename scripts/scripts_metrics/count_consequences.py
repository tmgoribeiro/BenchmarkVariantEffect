'''
Number of variants and positives per consequence class of one or more datasets.
'''
import argparse

import pandas as pd

from bend.utils.metrics_utils import read_table, find_consequence_column, infer_run, dataset_name, results_dir, save_results


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

        dataset = infer_run(path)[1] or dataset_name(path)
        log = {'file': path, 'n_variants': len(df), 'positives': int(df['label'].sum()), 'n_classes': len(counts),
               'classes': ','.join(map(str, counts.index))}
        save_results(results_dir(__file__, None, dataset), 'consequences', counts.reset_index(), [log])


if __name__ == '__main__':
    main()
