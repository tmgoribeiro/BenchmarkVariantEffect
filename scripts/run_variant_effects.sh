#!/bin/bash
# Runs the zero-shot variant effect prediction for every model on both BEND variant files.
# For each model, we extract the cosine distance between the reference and mutated sequence embedding.
# Run from the repository root. Results go to runs/<model>/<run_name>/.
#
# The embedding index defaults to the position of the variant in the window, so --embedding_idx
# only has to be set for DNABERT (v1), whose embeddings are per k-mer token.
# A model that fails does not stop the others.

GENOME=data/genomes/GRCh38.primary_assembly.genome.fa
CONTEXT=256        # nucleotides on each side of the variant (bidirectional models)
CAUSAL_CONTEXT=512 # nucleotides left of the variant (autoregressive models)
AG_CONTEXT=1024    # AlphaGenome needs windows whose length is a multiple of 2048

run_all () {
    VARIANT_FILE=$1
    PREFIX=$2

    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_dnabert_6 dnabert pretrained_models/dnabert/6-new-12w-0 $GENOME --extra_context $CONTEXT --embedding_idx $((CONTEXT - 2)) --kmer 6 # the 6-mer starting 2 nt before the variant has it at its 3rd position
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_dnabert2 dnabert2 zhihan1996/DNABERT-2-117M $GENOME --extra_context $CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_nt_1000g_500m nt InstaDeepAI/nucleotide-transformer-500m-1000g $GENOME --extra_context $CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_nt_1000g_25b nt InstaDeepAI/nucleotide-transformer-2.5b-1000g $GENOME --extra_context $CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_nt_ms_25b nt InstaDeepAI/nucleotide-transformer-2.5b-multi-species $GENOME --extra_context $CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_nt_href_500m nt InstaDeepAI/nucleotide-transformer-500m-human-ref $GENOME --extra_context $CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_nt_v2_500m nt InstaDeepAI/nucleotide-transformer-v2-500m-multi-species $GENOME --extra_context $CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_hyenadna_tiny1k hyenadna pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen $GENOME --extra_context $CAUSAL_CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_hyenadna_small32k hyenadna pretrained_models/hyenadna/hyenadna-small-32k-seqlen $GENOME --extra_context $CAUSAL_CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_hyenadna_medium160k hyenadna pretrained_models/hyenadna/hyenadna-medium-160k-seqlen $GENOME --extra_context $CAUSAL_CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_hyenadna_medium450k hyenadna pretrained_models/hyenadna/hyenadna-medium-450k-seqlen $GENOME --extra_context $CAUSAL_CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_hyenadna_large1m hyenadna pretrained_models/hyenadna/hyenadna-large-1m-seqlen $GENOME --extra_context $CAUSAL_CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_genalm_bert_large_t2t genalm AIRI-Institute/gena-lm-bert-large-t2t $GENOME --extra_context $CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_genalm_bigbird_base_t2t genalm AIRI-Institute/gena-lm-bigbird-base-t2t $GENOME --extra_context $CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_grover grover pretrained_models/grover $GENOME --extra_context $CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_borzoi borzoi johahi/borzoi-replicate-0 $GENOME --extra_context $CONTEXT
    python3 scripts/predict_variant_effects.py $VARIANT_FILE ${PREFIX}_ag ag pretrained_models/alphagenome/model_all_folds.safetensors $GENOME --extra_context $AG_CONTEXT
}

run_all data/variant_effects/variant_effects_expression.bed variant_effects_expression
run_all data/variant_effects/variant_effects_disease.bed variant_effects_disease
