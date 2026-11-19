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

    # The variant is at the 0-based position `start`, both in standard bed files (end = start + 1)
    # and in the TraitGym files written by prepare_traitgym.py (start = end).
    # Normalise the segments so that the windows are [start - left, start + right) for bidirectional models,
    # exactly 2 * extra_context long (AlphaGenome needs a length divisible by a power of 2), and
    # [start - left, start + 1) for autoregressive models, which have to end right after the variant.
    # Either way, the variant ends up at index extra_context_left.
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

    run_dir = Path(f"runs/{args.model}/{args.out_file}")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"embeddings_{args.out_file}.pkl"


    Ncount = 0
    start = time.perf_counter()


    with open(out_path, "wb") as f:
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

    #this is useful for alphagenome
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