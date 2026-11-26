'''
Same scoring as predict_variant_effects.py, with several poolings around the variant (cosine_distance, avg_n{n}, ccs_n{n}, cls).
Everything goes to results/pooling_strategies/<model>/<dataset>/<strategy>/. The embeddings at the variant position
(no pooling) are always saved, in variant/.
'''
import argparse
from bend.utils import embedders, Annotation
from bend.utils.metrics_utils import dataset_name, pooling_dir, embeddings_path, scores_path, score_column
from tqdm.auto import tqdm
from scipy import spatial
import time
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime
from contextlib import ExitStack


# Special tokens (leading, trailing) around the sequence of the models that have a [CLS] token.
# HyenaDNA is left out: it is causal, so its [CLS] sees none of the sequence.
CLS_MODELS = {
    'nt': (1, 0),        # [CLS] ... (no [SEP])
    'dnabert': (1, 1),   # [CLS] ... [SEP]
    'dnabert2': (1, 1),
    'grover': (1, 1),
}


def main():

    parser = argparse.ArgumentParser('Compute embeddings')
    parser.add_argument('bed_file', type=str, help='Path to the bed file')
    parser.add_argument('out_file', type=str, help='Run name, used in the names of the output files. Everything goes to results/pooling_strategies/<model>/<dataset>/')
    parser.add_argument('model', choices=['ag', 'nt', 'dnabert', 'hyenadna', 'dnabert2','grover','borzoi'], type=str, help='Model architecture for computing embeddings')
    parser.add_argument('checkpoint', type=str, help='Path to or name of the model checkpoint')
    parser.add_argument('genome', type=str, help='Path to the reference genome fasta file')
    parser.add_argument('--extra_context', type=int, default=1024, help='Number of extra nucleotides to include on each side of the sequence')
    parser.add_argument('--kmer', type=int, default=6, help = 'Kmer size for the DNABERT model')
    parser.add_argument('--embedding_idx', type=int, default=None, help = 'Index of the embedding to use for computing the distance. Defaults to the position of the variant in the window. Only needs to be set for models whose embeddings are not one per nucleotide (dnabert)')
    parser.add_argument('--neighbours', type=int, nargs='+', default=[2, 8, 32, 128], help='Radii (> 0) of the averaging windows around embedding_idx. The variant position alone is always saved, in variant/')
    parser.add_argument('--dataset', type=str, default=None, help='Name of the dataset folder. Defaults to the bed file name without the variant_effects_ prefix (variant_effects_disease.bed -> disease)')

    args = parser.parse_args()
    if any(n < 1 for n in args.neighbours):
        parser.error('--neighbours must be > 0: the variant position alone (no pooling) is always saved, in variant/')

    # out_file and dataset only name files and folders; a path in them would move the results elsewhere
    dataset = args.dataset if args.dataset is not None else dataset_name(args.bed_file)
    for arg_name, value in (('out_file', args.out_file), ('dataset', dataset)):
        if Path(value).name != value or value in ('', '.', '..'):
            parser.error(f'{arg_name} must be a plain name, not a path: {value!r}')

    # results/pooling_strategies/<model>/<dataset>/, anchored at the repository root so that it does not
    # depend on the directory the script is run from
    out_dir = pooling_dir(args.model, dataset)

    extra_context_left = args.extra_context
    extra_context_right = args.extra_context

    kwargs = {'disable_tqdm': True}
    # get the embedder
    if args.model == 'ag':
         embedder = embedders.AlphaGenomeEmbedder(args.checkpoint)
    elif args.model == 'nt':
         embedder = embedders.NucleotideTransformerEmbedder(args.checkpoint)
         kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    elif args.model == 'dnabert':
        embedder = embedders.DNABertEmbedder(args.checkpoint, kmer = args.kmer)
    elif args.model == 'hyenadna':
        embedder = embedders.HyenaDNAEmbedder(args.checkpoint)
        # autogressive model. No use for right context.
        extra_context_right = 0
    elif args.model == 'dnabert2':
        embedder = embedders.DNABert2Embedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    elif args.model == 'grover':
        embedder = embedders.GROVEREmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    elif args.model == 'borzoi':
        embedder = embedders.BorzoiEmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding

    has_cls = args.model in CLS_MODELS
    if has_cls:
        kwargs['remove_special_tokens'] = False # keep [CLS], it is removed from the per-position embeddings below
        n_leading, n_trailing = CLS_MODELS[args.model]


    # load the bed file
    genome_annotation = Annotation(args.bed_file, reference_genome=args.genome)

    # The variant is at the 0-based position `start`. Normalise the segments so that it ends up at
    # index extra_context_left of the window (see predict_variant_effects.py).
    causal = extra_context_right == 0
    genome_annotation.annotation['end'] = genome_annotation.annotation['start'] + int(causal)

    # extend the segments if necessary
    if args.extra_context > 0:
        genome_annotation.extend_segments(extra_context_left=extra_context_left, extra_context_right=extra_context_right)

    # position of the variant inside each window
    variant_idx = extra_context_left if args.extra_context > 0 else 0
    embedding_idx = args.embedding_idx if args.embedding_idx is not None else variant_idx
    print(f'Variant at index {variant_idx} of the window, using the embedding at index {embedding_idx}')

    genome_annotation.check_reference(variant_idx)

    # columns of the bed file, repeated in every strategy csv
    bed_columns = list(genome_annotation.annotation.columns)

    # one score per csv: cosine_distance, avg_n{n} and ccs_n{n} for every n, and cls
    scores = ['cosine_distance'] + [f'{s}_n{n}' for s in ('avg', 'ccs') for n in args.neighbours]
    if has_cls:
        scores.append('cls')
    else:
        print(f'{args.model} has no [CLS] token, the cls strategy is skipped')
    for score in scores:
        genome_annotation.annotation[score_column(score)] = np.nan

    # the embeddings at the variant go to variant/, the mean embeddings next to the avg_n{n} csv, the [CLS] embeddings next to the cls csv
    variant_path = embeddings_path(args.model, dataset, args.out_file, 'variant')
    out_paths = {n: embeddings_path(args.model, dataset, args.out_file, f'n{n}') for n in args.neighbours}
    cls_path = embeddings_path(args.model, dataset, args.out_file, 'cls')
    csv_paths = {score: scores_path(args.model, dataset, args.out_file, score) for score in scores}
    for path in [variant_path] + list(out_paths.values()) + list(csv_paths.values()) + ([cls_path] if has_cls else []):
        path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Results will be written to {out_dir}')

    Ncount = 0
    start = time.perf_counter()


    with ExitStack() as stack:
        variant_file = stack.enter_context(open(variant_path, "wb"))
        files = {r: stack.enter_context(open(p, "wb")) for r, p in out_paths.items()}
        if has_cls:
            cls_file = stack.enter_context(open(cls_path, "wb"))

        for index, row in tqdm(genome_annotation.annotation.iterrows()):

            dna = genome_annotation.get_dna_segment(index = index)

            # soft-masked genomes have lowercase repeats, which the tokenizers do not handle
            dna = dna.upper()
            Ncount += dna.count('N')
            dna = dna.replace('N', 'A')

            dna_alt = [x for x in dna]
            dna_alt[variant_idx] = row['alt']
            dna_alt = ''.join(dna_alt)

            emb_wt, emb_alt = embedder.embed([dna, dna_alt], **kwargs)
            emb_wt, emb_alt = np.asarray(emb_wt[0]), np.asarray(emb_alt[0])  # [L, D], plus the special tokens

            if has_cls:
                cls_wt, cls_alt = emb_wt[0], emb_alt[0]          # [D] each
                pickle.dump((index, cls_wt, cls_alt), cls_file, protocol=pickle.HIGHEST_PROTOCOL)
                genome_annotation.annotation.loc[index, 'distance_cls'] = spatial.distance.cosine(cls_alt, cls_wt)

                # keep only the per-position embeddings
                last = emb_wt.shape[0] - n_trailing
                emb_wt, emb_alt = emb_wt[n_leading:last], emb_alt[n_leading:last]

            # one embedding per nucleotide, so that embedding_idx is the variant (dnabert has one per k-mer)
            if args.model != 'dnabert' and emb_wt.shape[0] != len(dna):
                raise ValueError(f'{args.model} returned {emb_wt.shape[0]} embeddings for a sequence of {len(dna)} nucleotides')

            L = emb_wt.shape[0]

            d_pos = 1 - (emb_wt * emb_alt).sum(1) / (np.linalg.norm(emb_wt, axis=1) * np.linalg.norm(emb_alt, axis=1))
            genome_annotation.annotation.loc[index, 'distance'] = d_pos[embedding_idx]
            pickle.dump((index, emb_wt[embedding_idx], emb_alt[embedding_idx]), variant_file, protocol=pickle.HIGHEST_PROTOCOL)

            for n in args.neighbours:
                lo, hi = max(0, embedding_idx - n), min(L, embedding_idx + n + 1)
                wt_n, alt_n = emb_wt[lo:hi].mean(0), emb_alt[lo:hi].mean(0)

                pickle.dump((index, wt_n, alt_n), files[n], protocol=pickle.HIGHEST_PROTOCOL)

                genome_annotation.annotation.loc[index, f'distance_avg_n{n}'] = spatial.distance.cosine(alt_n, wt_n)
                genome_annotation.annotation.loc[index, f'distance_ccs_n{n}'] = d_pos[lo:hi].sum()


    elapsed = time.perf_counter() - start
    print(f"Total elapsed time: {elapsed}")
    print(f"N nucleotides replaced with A: {Ncount}")

    # one csv per score, each with the bed columns (start and end hold the window coordinates)
    for score, path in csv_paths.items():
        genome_annotation.annotation[bed_columns + [score_column(score)]].to_csv(path)

    labels = genome_annotation.annotation['label']
    auc_scores, auprc_scores = {}, {}
    for score in scores:
        distances = genome_annotation.annotation[score_column(score)].astype(float)
        auc_scores[score] = roc_auc_score(labels, distances)
        auprc_scores[score] = average_precision_score(labels, distances)
        print(f"AUC {score}: {auc_scores[score]:.4f} | AUPRC {score}: {auprc_scores[score]:.4f}")

    with open(out_dir / "log.txt", "a") as log_f:
        log_f.write(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} | "
            f"out_file={args.out_file} | "
            f"checkpoint={args.checkpoint} | "
            f"extra_context={args.extra_context} | "
            f"elapsed={elapsed:.2f}s | "
            f"n_variants={len(genome_annotation.annotation)} | "
            f"AUC={auc_scores} | "
            f"AUPRC={auprc_scores}\n"
        )


if __name__ == '__main__':
    main()
