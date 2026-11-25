# 🧬 Benchmarking DNA Language Models on Variant Effect Prediction

This repository benchmarks pretrained DNA language models (and sequence-to-function models) on non-coding variant effect prediction. It is built on top of the [BEND](https://github.com/frederikkemarin/BEND) codebase (Marin et al., ICLR 2024), reduced to the variant effect task and extended with:

- two new embedders: AlphaGenome (via `alphagenome-pytorch`) and Borzoi (via `borzoi-pytorch`);
- LoRA fine-tuning of DNABERT-2 with a contrastive loss on reference/alternative sequence pairs;
- per-run logging of the embeddings, cosine distances, AUROC and AUPRC;
- a helper script to annotate variants with the genes they fall in (GENCODE).

## How it works

For each variant in a bed file:

1. A window of the reference genome around the variant is extracted (`--extra_context` nucleotides on each side).
2. A second copy of the window is made with the reference base replaced by the alternative allele (alt).
3. Both sequences are embedded with the chosen model, and the embedding vector at the variant position (`--embedding_idx`) is taken from each.
4. The cosine distance between the two vectors is the variant's score: the larger the distance, the larger the predicted effect.
5. The scores are evaluated against the binary `label` column with AUROC and AUPRC.

No training is involved in the default (zero-shot) setting. Fine-tuning is optional; see [Fine-tuning](#fine-tuning-dnabert-2-with-lora-experimental).

## Repository structure

```
bend/
├── __init__.py                  # package marker (import bend.utils.embedders explicitly)
├── models/                      # model architectures loaded by the embedders
│   ├── dnabert2.py              # DNABERT-2 (local copy, no flash attention)
│   ├── dnabert2_padding.py      # padding helpers for dnabert2.py
│   └── hyena_dna.py             # HyenaDNA + character tokenizer
└── utils/
    ├── embedders.py             # one wrapper class per model, common .embed() interface
    ├── retrieve_from_bed.py     # Annotation: bed file + FASTA -> DNA sequences
    ├── download.py              # download helper for Zenodo-hosted checkpoints (GROVER)
    └── metrics_utils.py         # shared code of scripts/scripts_metrics (inputs, MLP, results/ folders)
scripts/
├── download_bend.py                             # download the BEND data share
├── prepare_traitgym.py                          # download a TraitGym dataset and convert it to a bed file
├── predict_variant_effects.py                   # zero-shot variant scoring (main script)
├── predict_variant_effects_pooling_strategies.py  # same scoring, several poolings; writes to results/pooling_strategies/
├── run_variant_effects.sh                       # runs every model on both BEND variant files
├── run_variant_effects_traitgym.sh              # runs every model on both TraitGym datasets (downloads them if needed)
├── run_pooling_strategies_traitgym.sh          # every pooling, classifier and ensemble on TraitGym
├── fine-tune_variant_effects.py                 # LoRA fine-tuning + test AUROC/AUPRC (DNABERT-2)
├── fine-tune_variant_effects_crossval.py        # k-fold version of the fine-tuning (validation loss only)
├── get_genes.py                                 # annotate expression variants with overlapping genes
└── scripts_metrics/                             # evaluation of the runs (see "Evaluation, classifiers and ensembles")
    ├── calculate_cosine_metrics.py              # AUROC/AUPRC of the distances, overall and per consequence
    ├── count_consequences.py                    # variants and positives per consequence class
    ├── confirm_genome.py                        # which assembly (hg19/hg38) a bed file uses
    ├── calculate_probabilities.py               # correlation of the distances with |logFC|
    ├── calculate_gene_metrics.py                # gene-level comparison with scRNA-seq variability
    ├── classifiers/                             # supervised models on the embeddings of one model
    │   ├── calculate_classifier_AUC.py          #   MLP, stratified k-fold
    │   ├── calculate_forest_classifier_AUC.py   #   random forest, stratified k-fold
    │   └── calculate_forest_gene_folds.py       #   random forest, chromosome holdout
    └── ensemble/                                # combinations of several models
        ├── strat_1_Cosine_average.py            #   mean of the distances (no training)
        ├── strat_2_Cosine_LG.py                 #   logistic regression on the distances
        ├── strat_3_PCA.py                       #   joint PCA of all embeddings + MLP
        ├── strat_4_Model_PCA.py                 #   one PCA per model + MLP
        ├── strat_5_Model_Clf_avg.py             #   one MLP per model, average of the predictions
        └── strat_6_Model_Clf_LG.py              #   one MLP per model, logistic regression on the logits
```

The scripts write to the git-ignored `results/` folder at the repository root, with one subfolder per script. The exception is `predict_variant_effects.py` (and the `run_variant_effects*.sh` scripts, which call it), which writes to the git-ignored `embeddings/<model>/<run_name>/`.

## Setup

The project targets **Python 3.10** (pinned to 3.10.13 in `.python-version`; `setup.py` accepts up to 3.11.9).

```bash
git clone https://github.com/tmgoribeiro/BenchmarkVariantEffect.git
cd BenchmarkVariantEffect
pip install -r requirements.txt
pip install -e .          # installs the `bend` package in editable mode
```

`transformers` is pinned to `<=4.33.3` because the local DNABERT-2 and HyenaDNA code depends on that API. `peft` is pinned to `0.10.0` for the same reason: newer versions import `transformers.Cache`, which only exists from transformers 4.36.
`bend/utils/embedders.py` imports `alphagenome_pytorch` and `peft` at module level, so both must be installed even if you only use another model.

Some scripts and models need packages that are **not** in `requirements.txt`:

| Needed for | Install |
| --- | --- |
| Borzoi embedder | `pip install --no-deps "borzoi-pytorch>=0.5.0" intervaltree sortedcontainers` (without `--no-deps`, pip upgrades `transformers` past the pinned version) |
| `scripts/download_bend.py` | `pip install requests beautifulsoup4 lxml` |
| `scripts/get_genes.py` | `pip install pyranges` |
| `scripts/scripts_metrics/calculate_gene_metrics.py` | `pip install scanpy scikit-misc` |
| `calculate_forest_gene_folds.py` on the GPU (optional) | cuML (RAPIDS); without it, scikit-learn is used |

**AlphaGenome needs a CUDA GPU**: the embedder moves the model and inputs to `cuda` explicitly. The other embedders fall back to CPU.

## Data

### Expected layout

Data, checkpoints and results are git-ignored. The scripts expect:

```
data/
├── genomes/
│   ├── GRCh38.primary_assembly.genome.fa
│   └── gencode.v50.genes_only.gtf          # only for get_genes.py
└── variant_effects/
    ├── variant_effects_expression.bed      # eQTLs (BEND, from DeepSEA)
    └── variant_effects_disease.bed         # disease variants (BEND, from ClinVar)
pretrained_models/                          # local checkpoints (e.g. alphagenome/, hyenadna/, grover/)
```

`python scripts/download_bend.py` downloads the BEND data share into the current directory.

### TraitGym

`scripts/prepare_traitgym.py` downloads a [TraitGym](https://huggingface.co/datasets/songlab/TraitGym) dataset from Hugging Face and writes it as a bed file:

```bash
python scripts/prepare_traitgym.py mendelian_traits_matched_9 data/variant_effects/traitgym_mendelian.bed data/genomes/GRCh38.primary_assembly.genome.fa
python scripts/prepare_traitgym.py complex_traits_matched_9   data/variant_effects/traitgym_complex.bed   data/genomes/GRCh38.primary_assembly.genome.fa
```

- Datasets: `mendelian_traits_matched_9`, `complex_traits_matched_9`, `mendelian_traits_all`, `complex_traits_all`. The `matched_9` sets have 9 benign variants per positive (3380 and 11400 variants, 10% positive).
- TraitGym positions are 1-based; the bed gets the 0-based position in `start` and `end = start`.
- The `chr` prefix is added when the FASTA uses it. The script stops if any variant's `ref` does not match the genome.
- `--chromosomes 19` keeps only some chromosomes, for a quick test.
- The other TraitGym columns (`consequence`, `match_group`, `tss_dist`, ...) are kept, so the metrics can be split by consequence.

### Variant file format

A tab-separated `bed` file with a header. `predict_variant_effects.py` needs at least these columns:

| Column | Meaning |
| --- | --- |
| `chromosome` | must match the FASTA record names (e.g. `chr1`) |
| `start`, `end` | 0-based position of the variant in `start`. Both `end = start + 1` (standard bed, BEND files) and `end = start` (TraitGym files) work |
| `alt` | alternative allele (single nucleotide) |
| `label` | 1 = functional / pathogenic, 0 = benign |
| `ref` | optional: reference allele, used to check the coordinates (see below) |

Any other columns are kept and copied to the output CSV.

`label` is required by `predict_variant_effects.py` and the pooling script (and so by the `.sh` scripts that call them): they compute the AUROC and AUPRC at the end. Without it, the embeddings and distances are saved, but the script then stops with an error. The classifier and ensemble scripts also need it.

If the file has a `ref` column, every script checks that the genome has that base at the variant position. It prints the number of mismatches and stops if more than 5% of the variants do not match. A wrong genome build or an off-by-one in the coordinates raises no other error; it only gives wrong embeddings.

When the genome path is exactly `data/genomes/GRCh37.no-chr.fa`, the `chr` prefix is stripped from the `chromosome` column so it matches that FASTA.

## Zero-shot variant effect prediction

```bash
python scripts/predict_variant_effects.py <bed_file> <run_name> <model> <checkpoint> <genome_fasta> \
    [--extra_context 1024] [--embedding_idx <variant position>] [--kmer 6]
```

Example:

```bash
python scripts/predict_variant_effects.py data/variant_effects/variant_effects_disease.bed \
    disease_dnabert2 dnabert2 zhihan1996/DNABERT-2-117M data/genomes/GRCh38.primary_assembly.genome.fa \
    --extra_context 1024
```

Three scripts run every model in one go. Run them from the repository root:

- `bash scripts/run_variant_effects.sh` runs 17 checkpoints on both BEND variant files.
- `bash scripts/run_variant_effects_traitgym.sh` runs DNABERT-2, GROVER, NT, HyenaDNA, Borzoi and AlphaGenome on the TraitGym Mendelian and complex-traits datasets. It downloads and converts the bed files first if they do not exist. All models get a 2048 bp window, and the run names are `tgm2048_<model>` and `tgc2048_<model>`. `PYTHON` and `GENOME` can be set from the environment.
- `bash scripts/run_pooling_strategies_traitgym.sh` runs the pooling script on the same models and datasets (cosine at the variant, average and CCS pooling over ±2 to ±128 nt, and [CLS]). It then trains the MLP and random forest on every pooling, and runs the six ensembles with all models (only DNABERT-2, GROVER and NT for [CLS]). It needs the bed files of `run_variant_effects_traitgym.sh`. `SKIP_EMBED=1` skips the embeddings and only reruns the metrics.

A model that fails does not stop the others.

### Windows and embedding index

The window around each variant is built so that the variant is always at index `extra_context`:

- **Bidirectional models** get `[start − extra_context, start + extra_context)`, exactly `2 × extra_context` nucleotides.
- **Autoregressive models** (`hyenadna`) get left context only: `[start − extra_context, start + 1)`, so the variant is the last nucleotide. These models are causal, so the embedding of any position before the variant is identical for both alleles. An index that is off by one would give a distance of exactly 0 for every variant.
- To give HyenaDNA the same 2048 bp window as the bidirectional models with `--extra_context 1024`, use `--extra_context 2047`. Use a checkpoint whose maximum length covers the window, such as `hyenadna-small-32k-seqlen`. `hyenadna-tiny-1k-seqlen` would split a 2048 bp window into two 1 kb chunks, so the variant would only see 1 kb of context.

`--embedding_idx` defaults to that position, so it only needs to be set for **DNABERT (v1)**. Its embeddings are not upsampled, so the index refers to k-mer tokens: use `extra_context − 2` to take the 6-mer that has the variant at its 3rd position.

**AlphaGenome** downsamples the sequence by powers of 2, so its window length must be a multiple of 2048. Use `--extra_context 1024` (or another multiple of 1024).

The window is converted to upper case (soft-masked genomes have lowercase repeats). Any `N` is then replaced with `A`, and the number of replaced `N`s is printed.

### Supported models

| `model` | Embedder class | Example checkpoint | Notes |
| --- | --- | --- | --- |
| `ag` | `AlphaGenomeEmbedder` | `pretrained_models/alphagenome/model_all_folds.safetensors` | CUDA only; 1 bp-resolution embeddings, human organism head |
| `borzoi` | `BorzoiEmbedder` | `johahi/borzoi-replicate-0` | one vector per 32 bp bin, upsampled to 1 bp |
| `dnabert2` | `DNABert2Embedder` | `zhihan1996/DNABERT-2-117M` | BPE, upsampled |
| `nt` | `NucleotideTransformerEmbedder` | `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species` | 6-mers, upsampled |
| `grover` | `GROVEREmbedder` | `pretrained_models/grover` | BPE, upsampled |
| `hyenadna` | `HyenaDNAEmbedder` | `pretrained_models/hyenadna/hyenadna-small-32k-seqlen` | autoregressive, use `--extra_context 2047` (see above); needs Git LFS for the download |
| `dnabert` | `DNABertEmbedder` | `pretrained_models/dnabert/6-new-12w-0` | use `--kmer` |

The embedders can also be used directly:

```python
from bend.utils.embedders import DNABert2Embedder

embedder = DNABert2Embedder('zhihan1996/DNABERT-2-117M')
embeddings = embedder.embed(['AGGATGCCGAGAGTATATGGGA'], upsample_embeddings=True)  # list of (1, length, dim) arrays
```

### Output

Each run writes to `embeddings/<model>/<run_name>/` at the repository root:

| File | Content |
| --- | --- |
| `embeddings_<run_name>.pkl` | one pickled tuple `(index, embedding_ref, embedding_alt)` per variant, written one after another |
| `cosine_distance_<run_name>.csv` | the input bed plus a `distance` column; `start` and `end` hold the window coordinates |
| `log.txt` | one line appended per run: timestamp, run name, elapsed time, number of variants, AUROC (`AUC=`) and AUPRC |

The `.pkl` file holds many sequential pickles, so read it in a loop:

```python
import pickle
records = []
with open('embeddings/dnabert2/disease_dnabert2/embeddings_disease_dnabert2.pkl', 'rb') as f:
    while True:
        try:
            records.append(pickle.load(f))
        except EOFError:
            break
```

## Pooling strategies

`scripts/predict_variant_effects_pooling_strategies.py` scores the variants in the same way as `predict_variant_effects.py`, with the same arguments, windows and embedding index. It also compares several ways of pooling the embeddings around the variant, and it saves the embeddings in the layout that the scripts in `scripts/scripts_metrics/` read. **Use this script when you want to run the evaluation, classifier or ensemble scripts.**

```bash
python scripts/predict_variant_effects_pooling_strategies.py <bed_file> <run_name> <model> <checkpoint> <genome_fasta> \
    [--extra_context 1024] [--neighbours 0 2 8 32 128] [--dataset <name>] [--embedding_idx <idx>] [--kmer 6]
```

| Strategy | Score |
| --- | --- |
| `cosine_distance` | cosine distance at the variant position (the score of `predict_variant_effects.py`) |
| `avg_n{n}` | cosine distance between the mean embeddings of positions `[idx − n, idx + n]` |
| `ccs_n{n}` | sum of the per-position cosine distances over `[idx − n, idx + n]` |
| `cls` | cosine distance between the `[CLS]` embeddings (only `nt`, `dnabert`, `dnabert2`, `grover`) |

`--neighbours` sets the radii `n`; 0 is the variant position alone.

`--dataset` names the dataset folder. It defaults to the bed file name without the `variant_effects_` prefix (`variant_effects_disease.bed` → `disease`).

Everything goes to `results/pooling_strategies/<model>/<dataset>/`. There is one folder per strategy, and inside `avg/` and `ccs/` one folder per `n`:

```
results/pooling_strategies/<model>/<dataset>/
├── cosine_distance/
│   └── cosine_distance_<run_name>.csv
├── avg/
│   ├── n0/
│   │   ├── avg_n0_<run_name>.csv
│   │   └── embeddings_<run_name>_n0.pkl
│   ├── n2/ ...
│   └── n128/ ...
├── ccs/
│   ├── n0/
│   │   └── ccs_n0_<run_name>.csv
│   └── ...
├── cls/                                  (models with [CLS] only)
│   ├── cls_<run_name>.csv
│   └── embeddings_<run_name>_cls.pkl
└── log.txt
```

- Every csv has the bed columns (including `label`) and that score's distance column (`distance`, `distance_avg_n8`, `distance_cls`, ...). `start` and `end` hold the window coordinates, not the variant position.
- `embeddings_<run_name>_n{n}.pkl` holds one `(index, ref, alt)` tuple per variant: the mean embedding over `[idx − n, idx + n]`. `_n0` is the variant position alone, the embedding that `cosine_distance` uses.
- `log.txt` gets one line per run, with the AUROC and AUPRC of every score.

Running the script again with the same `run_name` overwrites its csv and pkl files; only `log.txt` accumulates.

## Evaluation, classifiers and ensembles

The scripts in `scripts/scripts_metrics/` read the outputs of the pooling script in `results/pooling_strategies/<model>/<dataset>/`:

- The embeddings come from `avg/<pooling>/` (or `cls/`), chosen with `--pooling`.
- The labels come from `cosine_distance/cosine_distance_<run_name>.csv`.
- The distances come from the csv of the `--score`.

They write to `results/<script>/<model>/<dataset>/`. Ensembles write to `results/<script>/<model1+model2+...>/<dataset>/`. Every run writes two files there:

- `<run>.csv`: the full results, per fold or per class. It is overwritten when the same run is repeated.
- `log.txt`: one summary line appended per run. For the cross-validated scripts, the line has `AUC=mean+/-std` and `AUPRC=mean+/-std` over the folds, and `chance_AUPRC`, the fraction of positives (the AUPRC of a random classifier).

The code they share (paths, loading, the MLP) is in `bend/utils/metrics_utils.py`.

Print every summary at once:

```bash
find results -name log.txt -exec sh -c 'echo "== $1"; cat "$1"' _ {} \;
```

### Zero-shot metrics

```bash
python scripts/scripts_metrics/calculate_cosine_metrics.py results/pooling_strategies/*/<dataset>/cosine_distance/cosine_distance_*.csv
```

Gives the AUROC and AUPRC of every `distance*` column. The `all` row is the whole dataset, and there is one row per consequence class when the file has a `consequence` column. `--columns` restricts the columns.

The same command works for any other score. For example, `results/pooling_strategies/*/<dataset>/avg/n8/avg_n8_*.csv` gives the 8-neighbour average.

The model and dataset are read from the path of each csv; pass `--model` and `--dataset` for a csv outside `results/` and `embeddings/`.

### Classifiers on the embeddings of one model

```bash
python scripts/scripts_metrics/classifiers/calculate_classifier_AUC.py <model> <dataset> <run_name> [--pooling n0] [--features diff avg prod concat]
```

- `--pooling` selects the embeddings file: `n0`, `n8`, ... (in `avg/n{n}/`) or `cls`.
- `--features` sets how ref and alt are combined: `diff` = ref − alt, `avg`, `prod` or `concat`. Several values run once each.
- MLP options: `--folds 5 --epochs 10 --lr 1e-4 --hidden 256 --layers 1 --dropout 0.1 --seed 42`. `--hidden 0` gives a linear classifier, and `--random` skips training to give a baseline.
- `calculate_forest_classifier_AUC.py` takes the same arguments and trains a random forest (`--n_estimators`, `--max_depth`).
- `calculate_forest_gene_folds.py` holds out groups of chromosomes instead of random folds, so that nearby variants are never in both the training and test sets.
- `--embeddings` and `--labels` read any other pair of files, e.g. the output of `predict_variant_effects.py`.

### Ensembles of several models

All six scripts take the dataset and the models to combine:

```bash
python scripts/scripts_metrics/ensemble/<script>.py <dataset> --models dnabert2 grover hyenadna ag --out_files '<prefix>_{model}'
```

`--out_files` gives the run name of each model. It is either one name per model, in the order of `--models`, or a single template where `{model}` and `{dataset}` are replaced. The default is `{dataset}_{model}`. Quote the template so the shell does not touch the braces.

| Script | What is combined | Trained? | Main options |
| --- | --- | --- | --- |
| `strat_1_Cosine_average.py` | mean of the distances (raw and z-scored) | no | `--score cosine_distance` (or `avg_n{n}`, `ccs_n{n}`, `cls`) |
| `strat_2_Cosine_LG.py` | logistic regression on the distances | k-fold | `--score`, `--C`, `--folds` |
| `strat_3_PCA.py` | all embeddings, scaled, one joint PCA, MLP | k-fold | `--n_components 512`, `--pooling`, `--features` |
| `strat_4_Model_PCA.py` | one PCA per model, concatenated, MLP | k-fold | `--n_components 128`, `--pooling`, `--features` |
| `strat_5_Model_Clf_avg.py` | one MLP per model, mean of out-of-fold probabilities/logits | k-fold | `--pooling`, `--features` |
| `strat_6_Model_Clf_LG.py` | one MLP per model, logistic regression on their out-of-fold logits | k-fold | `--pooling`, `--features`, `--C` |

The embedding-based scripts (3 to 6) take the same MLP options as the classifiers. All models must have been run on the same variants.

`strat_4_Model_PCA.py` with a single model (`--models ag`) gives the "PCA + MLP" baseline of that model.

`strat_1` and `strat_5` report one AUROC/AUPRC over all the variants (all the out-of-fold predictions for `strat_5`), ± the std over the same 5 folds as the classifiers. The other k-fold scripts report mean ± std over the folds.

### Other metrics

| Script | Use |
| --- | --- |
| `count_consequences.py <bed or csv files>` | number of variants and positives per consequence class |
| `confirm_genome.py <bed> --genomes hg19=<fa> hg38=<fa>` | fraction of `ref` alleles that match each genome, to find the assembly of a bed |
| `calculate_probabilities.py <csv files>` | Spearman and Kendall correlations between the distances and `|logFC|` (`--effect_column`) |
| `calculate_gene_metrics.py <csv> --top 100` | top genes by distance vs. most variable genes in scRNA-seq (needs `gene_name`, from `get_genes.py`) |

### Using runs of `predict_variant_effects.py`

`predict_variant_effects.py` writes `embeddings/<model>/<run_name>/embeddings_<run_name>.pkl`. That file holds the variant-position embeddings, the same as `_n0`. The classifier scripts can read it with `--embeddings` and `--labels`. The ensemble scripts need the layout of the pooling script, so copy the files there first:

```bash
# embeddings/<model>/<run_name>/ -> results/pooling_strategies/<model>/<dataset>/
D=results/pooling_strategies/<model>/<dataset>
mkdir -p $D/avg/n0 $D/cosine_distance
cp embeddings/<model>/<run_name>/embeddings_<run_name>.pkl      $D/avg/n0/embeddings_<run_name>_n0.pkl
cp embeddings/<model>/<run_name>/cosine_distance_<run_name>.csv $D/cosine_distance/cosine_distance_<run_name>.csv
```

## Example: TraitGym from start to end

```bash
GENOME=data/genomes/GRCh38.primary_assembly.genome.fa

# 1. bed files
python scripts/prepare_traitgym.py mendelian_traits_matched_9 data/variant_effects/traitgym_mendelian.bed $GENOME
python scripts/prepare_traitgym.py complex_traits_matched_9   data/variant_effects/traitgym_complex.bed   $GENOME

# 2. embeddings and distances of each model (repeat for complex, with tgc_<model> and --dataset complex)
BED=data/variant_effects/traitgym_mendelian.bed
python scripts/predict_variant_effects_pooling_strategies.py $BED tgm_dnabert2 dnabert2 zhihan1996/DNABERT-2-117M $GENOME --dataset mendelian
python scripts/predict_variant_effects_pooling_strategies.py $BED tgm_grover   grover   pretrained_models/grover $GENOME --dataset mendelian
python scripts/predict_variant_effects_pooling_strategies.py $BED tgm_hyenadna hyenadna pretrained_models/hyenadna/hyenadna-small-32k-seqlen $GENOME --extra_context 2047 --dataset mendelian
python scripts/predict_variant_effects_pooling_strategies.py $BED tgm_ag       ag       pretrained_models/alphagenome/model_all_folds.safetensors $GENOME --dataset mendelian

# 3. zero-shot AUROC/AUPRC of each model
python scripts/scripts_metrics/calculate_cosine_metrics.py results/pooling_strategies/*/mendelian/cosine_distance/cosine_distance_*.csv

# 4. PCA + MLP of single models, then the ensembles
for m in dnabert2 ag; do
  python scripts/scripts_metrics/ensemble/strat_4_Model_PCA.py mendelian --models $m --out_files 'tgm_{model}'
done
for s in strat_1_Cosine_average strat_2_Cosine_LG strat_3_PCA strat_4_Model_PCA strat_5_Model_Clf_avg strat_6_Model_Clf_LG; do
  python scripts/scripts_metrics/ensemble/$s.py mendelian --models dnabert2 grover hyenadna ag --out_files 'tgm_{model}'
done

# 5. summaries
find results -name log.txt -exec sh -c 'echo "== $1"; cat "$1"' _ {} \;
```

On Windows PowerShell:

- Variables: `$GENOME = "..."`.
- Loops: `foreach ($m in 'dnabert2','ag') { ... }`.
- Wildcards: PowerShell does not expand `*` for Python, so pass `(Get-ChildItem results\pooling_strategies\*\mendelian\cosine_distance\cosine_distance_*.csv).FullName` in step 3.
- Summaries: `Get-ChildItem results -Recurse -Filter log.txt | ForEach-Object { "== $($_.Directory)"; Get-Content $_ }`.

## Fine-tuning DNABERT-2 with LoRA (experimental)

```bash
python scripts/fine-tune_variant_effects.py <bed_file> dnabert2 zhihan1996/DNABERT-2-117M <genome_fasta> <batch_size> <epochs> \
    [--extra_context 256] [--max_variants N]
```

- `--max_variants` uses only the first N variants, for quick tests. All variants are used by default.
- The window and embedding index are built as in the zero-shot script.
- Only DNABERT-2 can be fine-tuned (`DNABert2Embedder.prepare_for_finetuning` and `training_embed`), so `dnabert2` is the only model accepted.
- LoRA adapters (r = 8, alpha = 16, dropout = 0.1) are added to the `Wqkv` and `attention.output.dense` layers. Training uses AdamW with lr = 1e-5.
- The loss is contrastive on the cosine distance *d* between the reference and alternative embeddings: `label · max(0, margin − d)² + (1 − label) · d²`, with margin = 10. It pushes benign pairs together and functional pairs apart.
- The data is split 64 / 16 / 20 (train / validation / test), stratified by label with seed 42. After training, the test AUROC and AUPRC are computed with the same cosine distance score as the zero-shot script.

`fine-tune_variant_effects_crossval.py` runs the same training with `StratifiedKFold` (`--n_folds`, default 5) and reports the mean ± standard deviation of the best validation loss. It does not compute a test AUROC or AUPRC.

## Gene annotation

`scripts/get_genes.py` intersects `data/variant_effects/variant_effects_expression.bed` with the genes in `data/genomes/gencode.v50.genes_only.gtf` (using pyranges). It then:

- writes `data/variant_effects/variant_effects_expression_annotated.bed`, with one row per variant–gene overlap and `gene_id`/`gene_name` columns; intergenic variants are dropped;
- prints the 15 genes with the most variants.

## Known limitations

This is a work in progress:

- In the contrastive loss, the margin (10) is larger than the maximum cosine distance (2). The functional-variant term therefore never reaches zero, and the loss stays around 40–50.
- The fine-tuned LoRA weights are not saved, so they cannot be reused by the zero-shot scripts.
- The automatic HyenaDNA download uses Linux commands (`rm`, `mkdir -p`) and `git lfs`. On other systems, download the checkpoint manually into `pretrained_models/hyenadna/`.

## Acknowledgements and citation

The `bend` package, the baseline models and the variant effect datasets come from BEND:

    @inproceedings{marin2024bend,
      title={{BEND}: Benchmarking {DNA} Language Models on Biologically Meaningful Tasks},
      author={Frederikke Isa Marin and Felix Teufel and Marc Horlacher and Dennis Madsen and Dennis Pultz and Ole Winther and Wouter Boomsma},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=uKB4cFNQFg}
    }

Please also cite the original data sources:

- **Expression variants:** DeepSEA, [Zhou & Troyanskaya, *Nature Methods* 2015](https://www.nature.com/articles/nmeth.3547), with data from GRASP and the 1000 Genomes Project.
- **Disease variants:** ClinVar, [Landrum et al., *Nucleic Acids Research* 2019](https://doi.org/10.1093/nar/gkz972).
- **Gene annotation:** [GENCODE](https://www.gencodegenes.org/).

Also cite each model you use:

- [DNABERT](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680)
- [DNABERT-2](https://arxiv.org/abs/2306.15006)
- [Nucleotide Transformer](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2)
- [HyenaDNA](https://arxiv.org/abs/2306.15794)
- [GROVER](https://www.nature.com/articles/s42256-024-00872-0)
- [Borzoi](https://doi.org/10.1038/s41588-024-02053-6)
- [AlphaGenome](https://www.biorxiv.org/content/10.1101/2025.06.25.661532v1)

## License

See [LICENSE](LICENSE) (Creative Commons Attribution 4.0). The code in `bend/` is derived from [BEND](https://github.com/frederikkemarin/BEND); keep its original license notice when redistributing.
