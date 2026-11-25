'''
Zero-shot variant scoring: cosine distance between the ref and alt embeddings at the variant position.
Results go to embeddings/<model>/<run_name>/.
'''
import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import spatial
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from bend.utils import Annotation, embedders


def main():

    parser = argparse.ArgumentParser('Compute embeddings')
    parser.add_argument('bed_file', type=str, help='Path to the bed file')
    parser.add_argument('out_file', type=str, help='Run name, used in the names of the output files. Results go to embeddings/<model>/<out_file>/')
    parser.add_argument('model', choices=['ag', 'nt', 'dnabert', 'hyenadna', 'dnabert2','grover','borzoi'], type=str, help='Model architecture for computing embeddings')
    parser.add_argument('checkpoint', type=str, help='Path to or name of the model checkpoint')
    parser.add_argument('genome', type=str, help='Path to the reference genome fasta file')
    parser.add_argument('--extra_context', type=int, default=1024, help='Number of extra nucleotides to include on each side of the sequence')
    parser.add_argument('--kmer', type=int, default=6, help = 'Kmer size for the DNABERT model')
    parser.add_argument('--embedding_idx', type=int, default=None, help = 'Index of the embedding to use for computing the distance. Defaults to the position of the variant in the window. Only needs to be set for models whose embeddings are not one per nucleotide (dnabert)')

    args = parser.parse_args()

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
        extra_context_right = 0 # autoregressive model, no use for right context
    elif args.model == 'dnabert2':
        embedder = embedders.DNABert2Embedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    elif args.model == 'grover':
        embedder = embedders.GROVEREmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    elif args.model == 'borzoi':
        embedder = embedders.BorzoiEmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding

    # load the bed file
    genome_annotation = Annotation(args.bed_file, reference_genome=args.genome)

    if args.genome == 'data/genomes/GRCh37.no-chr.fa':
        ann = genome_annotation.annotation
        genome_annotation.annotation['chromosome'] = ann['chromosome'].astype(str).str.replace(r'^chr', '', regex=True)

    # Put the variant (0-based `start`) at index extra_context_left of every window.
    # Autoregressive models get left context only, so their window ends right after the variant.
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

    genome_annotation.annotation['distance'] = None

    # embeddings/<model>/<run name>/ at the repository root, whatever directory the script is run from
    run_dir = Path(__file__).resolve().parent.parent / 'embeddings' / args.model / args.out_file
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"embeddings_{args.out_file}.pkl"

    Ncount = 0
    start = time.perf_counter()

    with open(out_path, "wb") as f:
        for index, row in tqdm(genome_annotation.annotation.iterrows()):
            dna = genome_annotation.get_dna_segment(index = index)

            # soft-masked genomes have lowercase repeats, which the tokenizers do not handle
            dna = dna.upper()
            Ncount += dna.count('N')
            dna = dna.replace('N', 'A')

            dna_alt = [x for x in dna]
            dna_alt[variant_idx] = row['alt']
            dna_alt = ''.join(dna_alt)

            embedding_wt, embedding_alt = embedder.embed([dna, dna_alt], **kwargs)
            embedding_wt = embedding_wt[0, embedding_idx]
            embedding_alt = embedding_alt[0, embedding_idx]

            pickle.dump((index, np.asarray(embedding_wt), np.asarray(embedding_alt)), f,
                    protocol=pickle.HIGHEST_PROTOCOL)

            d = spatial.distance.cosine(embedding_alt, embedding_wt)
            genome_annotation.annotation.loc[index, 'distance'] = d

    elapsed = time.perf_counter() - start
    print(f"Total elapsed time: {elapsed}")
    print(f"N nucleotides replaced with A: {Ncount}")

    genome_annotation.annotation.to_csv(run_dir / f"cosine_distance_{args.out_file}.csv")

    labels = genome_annotation.annotation['label']
    distances = genome_annotation.annotation['distance'].astype(float)
    final_score = roc_auc_score(labels, distances)
    final_auprc = average_precision_score(labels, distances)
    print(f"Final Score: {final_score}")
    print(f"AUPRC: {final_auprc}")

    with open(run_dir / "log.txt", "a") as log_f:
        log_f.write(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} | "
            f"out_file={args.out_file} | "
            f"elapsed={elapsed:.2f}s | "
            f"n_variants={len(genome_annotation.annotation)} |"
            f"AUC={final_score} | "
            f"AUPRC={final_auprc}\n"
        )


if __name__ == '__main__':
    main()