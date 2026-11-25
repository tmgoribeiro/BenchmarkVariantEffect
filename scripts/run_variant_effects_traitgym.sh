#!/bin/bash
# Runs the zero-shot variant effect prediction (predict_variant_effects.py) for every model on the TraitGym datasets.
# For each model, we extract the cosine distance between the reference and mutated sequence embedding.
#
# Run from the repository root:
#
#     bash scripts/run_variant_effects_traitgym.sh
#
# PYTHON and GENOME can be overridden from the environment, e.g.
#     PYTHON=python GENOME=/data/GRCh38.fa bash scripts/run_variant_effects_traitgym.sh

PYTHON=${PYTHON:-python3}
GENOME=${GENOME:-data/genomes/GRCh38.primary_assembly.genome.fa}
BED_DIR=data/variant_effects
CONTEXT=1024         # nucleotides on each side of the variant (bidirectional models); a multiple of 1024
CAUSAL_CONTEXT=2047  # nucleotides left of the variant (autoregressive models), so that their window is also 2048 bp

# hyenadna-tiny-1k-seqlen would split the 2048 bp window into two 1 kb chunks, so the variant would only see 1 kb
HYENADNA_CHECKPOINT=pretrained_models/hyenadna/hyenadna-small-32k-seqlen

if [ ! -f "$GENOME" ]; then
    echo "Reference genome not found: $GENOME (set GENOME=<path to the GRCh38 fasta>)" >&2
    exit 1
fi

FAILED=()

# bed file of a TraitGym dataset, created only if it does not exist yet
prepare () {
    DATASET=$1
    BED=$2
    if [ -f "$BED" ]; then
        echo "=== $BED already exists"
    else
        echo "=== Downloading $DATASET -> $BED"
        $PYTHON scripts/prepare_traitgym.py $DATASET $BED $GENOME || { echo "Could not create $BED" >&2; exit 1; }
    fi
}

# one model on one bed file: run name, model, checkpoint, then any extra argument
run () {
    BED=$1
    RUN_NAME=$2
    echo "=== $RUN_NAME"
    $PYTHON scripts/predict_variant_effects.py $BED $RUN_NAME $3 $4 $GENOME "${@:5}" || FAILED+=("$RUN_NAME")
}

run_all () {
    BED=$1
    PREFIX=$2

    run $BED ${PREFIX}_dnabert2 dnabert2 zhihan1996/DNABERT-2-117M --extra_context $CONTEXT
    run $BED ${PREFIX}_grover   grover   pretrained_models/grover --extra_context $CONTEXT
    run $BED ${PREFIX}_nt       nt       InstaDeepAI/nucleotide-transformer-v2-500m-multi-species --extra_context $CONTEXT
    run $BED ${PREFIX}_hyenadna hyenadna $HYENADNA_CHECKPOINT --extra_context $CAUSAL_CONTEXT
    run $BED ${PREFIX}_borzoi   borzoi   johahi/borzoi-replicate-0 --extra_context $CONTEXT
    run $BED ${PREFIX}_ag       ag       pretrained_models/alphagenome/model_all_folds.safetensors --extra_context $CONTEXT  # needs a CUDA GPU
}

MENDELIAN=$BED_DIR/traitgym_mendelian.bed
COMPLEX=$BED_DIR/traitgym_complex.bed

prepare mendelian_traits_matched_9 $MENDELIAN
prepare complex_traits_matched_9 $COMPLEX

run_all $MENDELIAN tgm2048
run_all $COMPLEX tgc2048

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "=== All runs finished. Results in embeddings/<model>/<run_name>/"
else
    echo "=== Finished, but these runs failed: ${FAILED[*]}" >&2
    exit 1
fi
