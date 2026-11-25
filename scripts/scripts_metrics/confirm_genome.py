'''
Find which genome assembly a bed file uses, from the fraction of ref alleles that match each genome.
'''
import argparse
from pathlib import Path

import pandas as pd

from bend.utils import Annotation
from bend.utils.metrics_utils import REPO_ROOT, dataset_name, results_dir, save_results


DEFAULT_GENOMES = [f'hg19={REPO_ROOT / "data/genomes/GRCh37.no-chr.fa"}',
                   f'hg38={REPO_ROOT / "data/genomes/GRCh38.primary_assembly.genome.fa"}']


def check_genome(sample, genome_dict):
    '''Number of variants whose ref matches the genome, and number on chromosomes missing from it.'''
    prefix = 'chr' if any(name.startswith('chr') for name in genome_dict) else ''
    hits, missing = 0, 0
    for chrom, pos, ref in zip(sample['chromosome'], sample['start'], sample['ref']):
        chrom = prefix + chrom
        if chrom not in genome_dict:
            missing += 1
            continue
        hits += str(genome_dict[chrom].seq[int(pos)]).upper() == ref.upper()
    return hits, missing


def main():

    parser = argparse.ArgumentParser('Check which reference genome a bed file uses')
    parser.add_argument('bed_file', type=str, help='Bed file with chromosome, start, ref columns')
    parser.add_argument('--genomes', type=str, nargs='+', default=DEFAULT_GENOMES, help='Genomes to test, as name=path.fa. Default hg19 and hg38 in data/genomes/')
    parser.add_argument('--n', type=int, default=2000, help='Number of variants sampled')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    genomes = []
    for value in args.genomes:
        name, sep, path = value.partition('=')
        if not sep:
            parser.error(f'--genomes values must be name=path, got {value!r}')
        if not Path(path).exists():
            parser.error(f'File not found: {path}')
        genomes.append((name, path))

    bed = pd.read_csv(args.bed_file, sep='\t')
    bed['chromosome'] = bed['chromosome'].astype(str).str.replace('chr', '', regex=False)
    sample = bed.sample(min(args.n, len(bed)), random_state=args.seed)
    print(f'{len(sample)} of {len(bed)} variants of {args.bed_file}')

    rows = []
    for name, path in genomes:
        hits, missing = check_genome(sample, Annotation(reference_genome=path).genome_dict)
        checked = len(sample) - missing
        rows.append({'genome': name, 'path': path, 'checked': checked, 'missing_chromosome': missing,
                     'hits': hits, 'hit_fraction': hits / checked if checked else float('nan')})
        print(f"{name}: {rows[-1]['hit_fraction']:.4f} of the ref alleles match ({hits}/{checked}, "
              f'{missing} variants on chromosomes missing from the genome)')

    log = {'bed_file': args.bed_file, 'n': len(sample), 'seed': args.seed,
           **{r['genome']: f"{r['hit_fraction']:.4f}" for r in rows}}
    save_results(results_dir(__file__, None, dataset_name(args.bed_file)), 'genome_check', pd.DataFrame(rows), [log])


if __name__ == '__main__':
    main()
