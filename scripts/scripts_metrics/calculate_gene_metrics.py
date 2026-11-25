'''
Compare the genes with the largest mean distance with the most variable genes of a scRNA-seq dataset (hypergeometric test).
Needs scanpy and scikit-misc.
'''
import argparse
from pathlib import Path

import scanpy as sc
from scipy.stats import hypergeom

from bend.utils.metrics_utils import read_table, resolve_run, results_dir, save_results


def main():

    parser = argparse.ArgumentParser('Overlap between the genes with the largest distance and the most variable genes in scRNA-seq')
    parser.add_argument('file', type=str, help='csv written by a prediction script on a bed annotated with gene_name (scripts/get_genes.py)')
    parser.add_argument('--score', type=str, default='distance', help='Distance column. Default distance')
    parser.add_argument('--top', type=int, default=100, help='Size of each top list')
    parser.add_argument('--n_top_genes', type=int, default=2000, help='Highly variable genes computed by scanpy')
    parser.add_argument('--min_cells', type=int, default=3, help='Genes expressed in fewer cells are discarded')
    parser.add_argument('--model', type=str, default=None, help='Model folder of the results. Defaults to the model in the path of the csv')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset folder of the results. Defaults to the dataset in the path of the csv')

    args = parser.parse_args()

    model, dataset = resolve_run(parser, args.file, args.model, args.dataset)
    df = read_table(args.file)
    for column in ('gene_name', args.score):
        if column not in df.columns:
            parser.error(f'No column {column} in {args.file}')

    genes = df.groupby('gene_name').agg(distance=(args.score, 'mean'), label=('label', 'max'))

    adata = sc.datasets.pbmc3k()
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_top_genes, flavor='seurat_v3')

    common = sorted(set(genes.index) & set(adata.var_names))
    print(f'Sequence data: {len(genes)} genes')
    print(f'scRNA-seq data: {adata.n_vars} genes')
    print(f'Common: {len(common)} genes')

    table = genes.loc[common].assign(variances_norm=adata.var['variances_norm'].loc[common])
    top_dist = set(table['distance'].nlargest(args.top).index)
    top_var = set(table['variances_norm'].nlargest(args.top).index)
    table['top_distance'] = table.index.isin(top_dist)
    table['top_variance'] = table.index.isin(top_var)

    overlap = len(top_dist & top_var)
    # P(overlap >= observed) when drawing len(top_dist) genes out of the shared genes, len(top_var) of them "successes"
    pval = hypergeom.sf(overlap - 1, len(common), len(top_var), len(top_dist))
    print(f'Overlap: {overlap}/{args.top}, p-value: {pval:.4g}')

    log = {'file': args.file, 'score': args.score, 'top': args.top, 'n_top_genes': args.n_top_genes,
           'min_cells': args.min_cells, 'common_genes': len(common), 'overlap': overlap, 'p_value': f'{pval:.4g}'}
    save_results(results_dir(__file__, model, dataset), Path(args.file).stem,
                 table.sort_values('distance', ascending=False).reset_index(), [log])


if __name__ == '__main__':
    main()
