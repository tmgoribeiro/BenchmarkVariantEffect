'''
This script differs from the default precompute_embeddings.py script in that it
computes embeddings for two sequences: the reference sequence and the variant
sequence. The variant sequence is obtained by replacing the reference nucleotide
with the variant nucleotide at the variant position.
'''
import argparse
from bend.utils import embedders, Annotation
from tqdm.auto import tqdm
from scipy import spatial
import time
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime
from contextlib import ExitStack



def main():

    parser = argparse.ArgumentParser('Compute embeddings')
    parser.add_argument('bed_file', type=str, help='Path to the bed file')
    parser.add_argument('out_file', type=str, help='Path to the output file')
    # model can be any of the ones supported by bend.utils.embedders
    parser.add_argument('model', choices=['ag', 'nt', 'dnabert', 'gpn', 'genalm', 'hyenadna', 'dnabert2','grover','borzoi'], type=str, help='Model architecture for computing embeddings')
    parser.add_argument('checkpoint', type=str, help='Path to or name of the model checkpoint')
    parser.add_argument('genome', type=str, help='Path to the reference genome fasta file')
    parser.add_argument('--extra_context', type=int, default=1024, help='Number of extra nucleotides to include on each side of the sequence')
    parser.add_argument('--kmer', type=int, default=6, help = 'Kmer size for the DNABERT model')
    parser.add_argument('--embedding_idx', type=int, default=None, help = 'Index of the embedding to use for computing the distance. Defaults to the position of the variant in the window. Only needs to be set for models whose embeddings are not one per nucleotide (dnabert)')
    parser.add_argument('--neighbours', type=int, nargs='+', default=[0, 2, 8, 32, 128], help='Radii of the averaging window around embedding_idx (0 = single position)')

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
    elif args.model == 'gpn':
        embedder = embedders.GPNEmbedder(args.checkpoint)
    elif args.model == 'genalm':
        embedder = embedders.GENALMEmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    elif args.model == 'hyenadna':
        embedder = embedders.HyenaDNAEmbedder(args.checkpoint)
        # autogressive model. No use for right context.
        extra_context_left = args.extra_context
        extra_context_right = 0
    elif args.model == 'dnabert2':
        embedder = embedders.DNABert2Embedder(args.checkpoint)#, lora = "pretrained_models/DNABert2_lora")
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    elif args.model == 'grover':
        embedder = embedders.GROVEREmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    elif args.model == 'borzoi':
        embedder = embedders.BorzoiEmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    else:
        raise ValueError('Model not supported')
    

    # load the bed file
    genome_annotation = Annotation(args.bed_file, reference_genome=args.genome)

    if args.genome == 'data/genomes/GRCh37.no-chr.fa':
        ann = genome_annotation.annotation
        genome_annotation.annotation['chromosome'] = ann['chromosome'].astype(str).str.replace(r'^chr', '', regex=True)

    #genome_annotation.annotation = genome_annotation.annotation.iloc[20000:30000].reset_index(drop=True)

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

    for n in args.neighbours:
        genome_annotation.annotation[f'distance_avg_n{n}'] = np.nan
        genome_annotation.annotation[f'distance_ccs_n{n}'] = np.nan
    genome_annotation.annotation['distance_cls'] = np.nan

    run_dir = Path(f"runs/{args.model}/{args.out_file}")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {n: run_dir / f"embeddings_{args.out_file}_n{n}.pkl" for n in args.neighbours}

    Ncount = 0
    start = time.perf_counter()


    with ExitStack() as stack:
        files = {r: stack.enter_context(open(p, "wb")) for r, p in out_paths.items()}
        cls_file = stack.enter_context(open(run_dir / f"embeddings_{args.out_file}_cls.pkl", "wb"))

        for index, row in tqdm(genome_annotation.annotation.iterrows()):

            # middle_point = row['start'] + 256
            # index the right embedding with dna[len(dna)//2]
            dna = genome_annotation.get_dna_segment(index = index)

            # soft-masked genomes have lowercase repeats, which the tokenizers do not handle
            dna = dna.upper()
            Ncount += dna.count('N')
            dna = dna.replace('N', 'A')

            dna_alt = [x for x in dna]
            dna_alt[variant_idx] = row['alt']
            dna_alt = ''.join(dna_alt)

            emb_wt, emb_alt = embedder.embed([dna, dna_alt], remove_special_tokens=False, **kwargs)
            emb_wt, emb_alt = np.asarray(emb_wt[0]), np.asarray(emb_alt[0])  # [L, D]

            cls_wt, cls_alt = emb_wt[0], emb_alt[0]          # [D] each
            emb_wt, emb_alt = emb_wt[1:-1], emb_alt[1:-1]
            pickle.dump((index, cls_wt, cls_alt), cls_file, protocol=pickle.HIGHEST_PROTOCOL)
            genome_annotation.annotation.loc[index, 'distance_cls'] = spatial.distance.cosine(cls_alt, cls_wt)

            L = emb_wt.shape[0]

            d_pos = 1 - (emb_wt * emb_alt).sum(1) / (np.linalg.norm(emb_wt, axis=1) * np.linalg.norm(emb_alt, axis=1))

            for n in args.neighbours:
                lo, hi = max(0, embedding_idx - n), min(L, embedding_idx + n + 1)
                wt_n, alt_n = emb_wt[lo:hi].mean(0), emb_alt[lo:hi].mean(0)

                pickle.dump((index, wt_n, alt_n), files[n], protocol=pickle.HIGHEST_PROTOCOL)

                genome_annotation.annotation.loc[index, f'distance_avg_n{n}'] = spatial.distance.cosine(alt_n, wt_n)
                genome_annotation.annotation.loc[index, f'distance_ccs_n{n}'] = d_pos[lo:hi].sum()


    elapsed = time.perf_counter() - start
    print(f"Total elapsed time: {elapsed}")
    print(f"N nucleotides replaced with A: {Ncount}")

    #this is useful for alphagenome
    genome_annotation.annotation.to_csv(run_dir / f"cosine_distance_{args.out_file}.csv")

    labels = genome_annotation.annotation['label']
    scores, auprc_scores = {}, {}
    for name in [f'{kind}_n{n}' for n in args.neighbours for kind in ('avg', 'ccs')] + ['cls']:
        distances = genome_annotation.annotation[f'distance_{name}'].astype(float)
        scores[name] = roc_auc_score(labels, distances)
        auprc_scores[name] = average_precision_score(labels, distances)
        print(f"AUC {name}: {scores[name]:.4f} | AUPRC {name}: {auprc_scores[name]:.4f}")

    with open(run_dir / "log.txt", "a") as log_f:
        log_f.write(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} | "
            f"out_file={args.out_file} | "
            f"elapsed={elapsed:.2f}s | "
            f"n_variants={len(genome_annotation.annotation)} | "
            f"AUC={scores} | "
            f"AUPRC={auprc_scores}\n"
        )


if __name__ == '__main__':
    main()