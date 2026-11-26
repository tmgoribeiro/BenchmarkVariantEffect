#!/bin/bash
# Every pooling strategy and every metric for every model on the TraitGym datasets:
#   1. predict_variant_effects_pooling_strategies.py: cosine at the variant, avg_n{n}, ccs_n{n} and cls (zero-shot)
#   2. MLP and random forest on the embeddings of each model: at the variant (no pooling), n{n} and cls
#   3. the six ensembles, with all the models (only the models with a [CLS] token for cls)
#
# Run from the repository root, after scripts/run_variant_effects_traitgym.sh has created the bed files:
#
#     bash scripts/run_pooling_strategies_traitgym.sh
#
# SKIP_EMBED=1 skips step 1 (embeddings already in results/pooling_strategies/). PYTHON and GENOME can be
# overridden from the environment, as in run_variant_effects_traitgym.sh.

PYTHON=${PYTHON:-python3}
GENOME=${GENOME:-data/genomes/GRCh38.primary_assembly.genome.fa}
BED_DIR=data/variant_effects
CONTEXT=1024
CAUSAL_CONTEXT=2047
NEIGHBOURS="2 8 32 128"

MODELS="dnabert2 grover hyenadna ag nt borzoi"
CLS_MODELS="dnabert2 grover nt"   # the models with a [CLS] token
POOLINGS="variant n2 n8 n32 n128"   # variant: the variant position alone, no pooling
SCORES="cosine_distance avg_n2 avg_n8 avg_n32 avg_n128 ccs_n2 ccs_n8 ccs_n32 ccs_n128"
METRICS=scripts/scripts_metrics
FAILED=()

checkpoint () {
    case $1 in
        dnabert2) echo zhihan1996/DNABERT-2-117M ;;
        grover)   echo pretrained_models/grover ;;
        nt)       echo InstaDeepAI/nucleotide-transformer-v2-500m-multi-species ;;
        hyenadna) echo pretrained_models/hyenadna/hyenadna-small-32k-seqlen ;;
        borzoi)   echo johahi/borzoi-replicate-0 ;;
        ag)       echo pretrained_models/alphagenome/model_all_folds.safetensors ;;  # needs a CUDA GPU
    esac
}

# runs a command, and remembers its name if it fails
try () {
    NAME=$1
    echo "=== $NAME"
    "${@:2}" || FAILED+=("$NAME")
}

run_dataset () {
    DATASET=$1   # results folder: mendelian or complex
    PREFIX=$2    # run name prefix: tgm2048 or tgc2048
    BED=$BED_DIR/traitgym_$DATASET.bed

    if [ -z "$SKIP_EMBED" ]; then
        [ -f "$BED" ] || { echo "Bed file not found: $BED (run scripts/run_variant_effects_traitgym.sh first)" >&2; exit 1; }
        [ -f "$GENOME" ] || { echo "Reference genome not found: $GENOME (set GENOME=<path to the GRCh38 fasta>)" >&2; exit 1; }
        for MODEL in $MODELS; do
            CONTEXT_MODEL=$CONTEXT
            [ "$MODEL" = hyenadna ] && CONTEXT_MODEL=$CAUSAL_CONTEXT
            try "pooling ${PREFIX}_$MODEL" $PYTHON scripts/predict_variant_effects_pooling_strategies.py \
                $BED ${PREFIX}_$MODEL $MODEL $(checkpoint $MODEL) $GENOME \
                --extra_context $CONTEXT_MODEL --neighbours $NEIGHBOURS --dataset $DATASET
        done
    fi

    # single models: MLP, random forest and PCA + MLP on every pooling
    for MODEL in $MODELS; do
        P_MODEL=$POOLINGS
        [[ " $CLS_MODELS " == *" $MODEL "* ]] && P_MODEL="$POOLINGS cls"
        for POOLING in $P_MODEL; do
            try "mlp $DATASET $MODEL $POOLING" $PYTHON $METRICS/classifiers/calculate_classifier_AUC.py $MODEL $DATASET ${PREFIX}_$MODEL --pooling $POOLING
            try "rf $DATASET $MODEL $POOLING" $PYTHON $METRICS/classifiers/calculate_forest_classifier_AUC.py $MODEL $DATASET ${PREFIX}_$MODEL --pooling $POOLING
            try "pca $DATASET $MODEL $POOLING" $PYTHON $METRICS/ensemble/strat_4_Model_PCA.py $DATASET --models $MODEL --out_files "${PREFIX}_{model}" --pooling $POOLING
        done
    done

    # ensembles of the distances: all the models, and the [CLS] models for cls
    for SCORE in $SCORES cls; do
        ENS=$MODELS
        [ "$SCORE" = cls ] && ENS=$CLS_MODELS
        for STRAT in strat_1_Cosine_average strat_2_Cosine_LG; do
            try "$STRAT $DATASET $SCORE" $PYTHON $METRICS/ensemble/$STRAT.py $DATASET --models $ENS --out_files "${PREFIX}_{model}" --score $SCORE
        done
    done

    # ensembles of the embeddings
    for POOLING in $POOLINGS cls; do
        ENS=$MODELS
        [ "$POOLING" = cls ] && ENS=$CLS_MODELS
        for STRAT in strat_3_PCA strat_4_Model_PCA strat_5_Model_Clf_avg strat_6_Model_Clf_LG; do
            try "$STRAT $DATASET $POOLING" $PYTHON $METRICS/ensemble/$STRAT.py $DATASET --models $ENS --out_files "${PREFIX}_{model}" --pooling $POOLING
        done
    done
}

run_dataset mendelian tgm2048
run_dataset complex tgc2048

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "=== All runs finished. Results in results/<script>/<model>/<dataset>/"
else
    echo "=== Finished, but these runs failed: ${FAILED[*]}" >&2
    exit 1
fi
