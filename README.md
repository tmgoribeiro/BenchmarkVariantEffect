# 🧬 Benchmarking DNA Language Models on Variant Effect Prediction

This repository benchmarks pretrained DNA language models (and sequence-to-function models) on non-coding variant effect prediction. It is built on the [BEND](https://github.com/frederikkemarin/BEND) codebase (Marin et al., ICLR 2024), reduced to the variant effect task and extended with AlphaGenome and Borzoi embedders, TraitGym support, pooling strategies, classifiers and ensembles of several models.

## How it works

For each variant in a bed file:

1. A window of the reference genome around the variant is extracted, and a copy is made with the alternative allele.
2. Both sequences are embedded with the chosen model, and the embedding at the variant position is taken from each.
3. The cosine distance between the two embeddings is the variant's score (zero-shot, no training).
4. The scores are evaluated against the `label` column with AUROC and AUPRC.

The embeddings can then be used to train classifiers or to combine several models (see [Classifiers and ensembles](#classifiers-and-ensembles)).

## Repository structure

```
bend/
├── models/                      # DNABERT-2 and HyenaDNA architectures
└── utils/
    ├── embedders.py             # one wrapper per model, common .embed() interface
    ├── retrieve_from_bed.py     # bed file + FASTA -> DNA sequences
    └── metrics_utils.py         # code shared by scripts/scripts_metrics
scripts/
├── download_bend.py                               # download the BEND data
├── prepare_traitgym.py                            # download a TraitGym dataset as a bed file
├── predict_variant_effects.py                     # zero-shot scoring, embedding at the variant
├── predict_variant_effects_pooling_strategies.py  # same scoring with several poolings, input of the metric scripts
├── run_variant_effects.sh                         # every model on the BEND variant files
├── run_variant_effects_traitgym.sh                # every model on TraitGym
├── run_pooling_strategies_traitgym.sh             # every pooling, classifier and ensemble on TraitGym
├── get_genes.py                                   # annotate variants with the genes they fall in
└── scripts_metrics/
    ├── calculate_cosine_metrics.py      # AUROC/AUPRC of the distances, overall and per consequence
    ├── classifiers/                     # MLP and random forest on the embeddings of one model
    ├── ensemble/                        # strat_1 to strat_6: combinations of several models
    └── ...                              # other metrics (consequence counts, genome check, |logFC| correlation)
```

Outputs go to the git-ignored `results/` (and `embeddings/` for `predict_variant_effects.py`).

## Setup

```bash
git clone https://github.com/tmgoribeiro/BenchmarkVariantEffect.git
cd BenchmarkVariantEffect
pip install -r requirements.txt
pip install -e .
```

- Python 3.10–3.11. `transformers` is pinned to `<=4.33.3` because the local DNABERT-2 and HyenaDNA code needs that API.
- Borzoi: `pip install --no-deps "borzoi-pytorch>=0.5.0" intervaltree sortedcontainers` (without `--no-deps`, pip upgrades `transformers`).
- AlphaGenome needs a CUDA GPU. The other models also run on CPU.
- Optional: `requests beautifulsoup4 lxml` for `download_bend.py`, `pyranges` for `get_genes.py`, `scanpy scikit-misc` for `calculate_gene_metrics.py`.

## Data

Data and checkpoints are git-ignored. The scripts expect the genome in `data/genomes/` (e.g. `GRCh38.primary_assembly.genome.fa`), the bed files in `data/variant_effects/` and local checkpoints in `pretrained_models/`. `python scripts/download_bend.py` downloads the BEND variant files.

**TraitGym:**

```bash
python scripts/prepare_traitgym.py mendelian_traits_matched_9 data/variant_effects/traitgym_mendelian.bed <genome_fasta>
python scripts/prepare_traitgym.py complex_traits_matched_9   data/variant_effects/traitgym_complex.bed   <genome_fasta>
```

**Bed format** (tab-separated, with a header), for any other dataset:

| Column | Meaning |
| --- | --- |
| `chromosome` | must match the FASTA names (`chr1` or `1`) |
| `start`, `end` | 0-based position of the variant in `start` (`end` = `start` or `start + 1`) |
| `alt` | alternative allele (single nucleotide) |
| `label` | 1 = functional / pathogenic, 0 = benign. **Required**: without it the scripts save the embeddings and then stop with an error when computing the AUROC |
| `ref` | optional, but recommended: the scripts stop if more than 5% of the `ref` alleles do not match the genome (wrong build or coordinates) |

Other columns are kept in the output csv. Windows that reach past the start or end of a chromosome are filled with `N` (embedded as `A`), so the variant keeps its position in the window.

## Zero-shot variant effect prediction

```bash
python scripts/predict_variant_effects.py <bed_file> <run_name> <model> <checkpoint> <genome_fasta> [--extra_context 1024]
```

Example:

```bash
python scripts/predict_variant_effects.py data/variant_effects/traitgym_mendelian.bed \
    tgm_dnabert2 dnabert2 zhihan1996/DNABERT-2-117M data/genomes/GRCh38.primary_assembly.genome.fa --extra_context 1024
```

| `model` | Example checkpoint | `--extra_context` (2048 bp window) |
| --- | --- | --- |
| `dnabert2` | `zhihan1996/DNABERT-2-117M` | 1024 |
| `grover` | `pretrained_models/grover` | 1024 |
| `nt` | `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species` | 1024 |
| `borzoi` | `johahi/borzoi-replicate-0` | 1024 |
| `ag` | `pretrained_models/alphagenome/model_all_folds.safetensors` | 1024 (window must be a multiple of 2048; CUDA only) |
| `hyenadna` | `pretrained_models/hyenadna/hyenadna-small-32k-seqlen` | 2047 (causal, left context only) |
| `dnabert` | `pretrained_models/dnabert/6-new-12w-0` | also set `--kmer` and `--embedding_idx extra_context − 2` |

`--extra_context` is the number of nucleotides on each side of the variant. HyenaDNA only sees the left side, so it gets 2047 to have the same 2048 bp window.

Each run writes to `embeddings/<model>/<run_name>/`:

- `embeddings_<run_name>.pkl`: one `(index, embedding_ref, embedding_alt)` tuple per variant, pickled one after another (read with `pickle.load` in a loop until `EOFError`);
- `cosine_distance_<run_name>.csv`: the bed plus a `distance` column;
- `log.txt`: AUROC and AUPRC of each run.

`run_variant_effects.sh` (BEND) and `run_variant_effects_traitgym.sh` (TraitGym) run every model in one go.

## Pooling strategies

`predict_variant_effects_pooling_strategies.py` takes the same arguments and also pools the embeddings around the variant. **Use it when you want to run the classifiers or ensembles.**

```bash
python scripts/predict_variant_effects_pooling_strategies.py <bed_file> <run_name> <model> <checkpoint> <genome_fasta> \
    [--extra_context 1024] [--neighbours 2 8 32 128] [--dataset <name>]
```

| Strategy | Score |
| --- | --- |
| `cosine_distance` | cosine distance at the variant |
| `avg_n{n}` | cosine distance between the mean embeddings over ±n nucleotides |
| `ccs_n{n}` | sum of the per-position cosine distances over ±n nucleotides |
| `cls` | cosine distance between the `[CLS]` embeddings (`nt`, `dnabert`, `dnabert2`, `grover`) |

Everything goes to `results/pooling_strategies/<model>/<dataset>/`: one csv per score and the embeddings: at the variant position, without pooling (`variant/embeddings_<run_name>_variant.pkl`, always saved), averaged over ±n (`avg/n{n}/embeddings_<run_name>_n{n}.pkl`) and `[CLS]` (`cls/embeddings_<run_name>_cls.pkl`). `--dataset` names that folder.

## Classifiers and ensembles

The scripts in `scripts/scripts_metrics/` read `results/pooling_strategies/` and write to `results/<script>/<model>/<dataset>/`, with a csv per run and a `log.txt` with the AUROC and AUPRC (mean ± std over 5 stratified folds for the trained ones).

```bash
# zero-shot AUROC/AUPRC, overall and per consequence
python scripts/scripts_metrics/calculate_cosine_metrics.py results/pooling_strategies/*/<dataset>/cosine_distance/*.csv

# MLP or random forest on the embeddings of one model (--pooling variant (default, no pooling), n8, ..., cls)
python scripts/scripts_metrics/classifiers/calculate_classifier_AUC.py <model> <dataset> <run_name> [--pooling variant]
python scripts/scripts_metrics/classifiers/calculate_forest_classifier_AUC.py <model> <dataset> <run_name> [--pooling variant]

# ensemble of several models; '<prefix>_{model}' gives the run name of each model
python scripts/scripts_metrics/ensemble/<strategy>.py <dataset> --models dnabert2 grover hyenadna ag --out_files '<prefix>_{model}'
```

| Ensemble | What is combined |
| --- | --- |
| `strat_1_Cosine_average` | mean of the distances (no training) |
| `strat_2_Cosine_LG` | logistic regression on the distances |
| `strat_3_PCA` | one PCA of all the embeddings, then an MLP |
| `strat_4_Model_PCA` | one PCA per model, then an MLP (with one model: the PCA + MLP baseline) |
| `strat_5_Model_Clf_avg` | one MLP per model, mean of their predictions |
| `strat_6_Model_Clf_LG` | one MLP per model, logistic regression on their predictions |

All models in an ensemble must have been run on the same bed file. `bash scripts/run_pooling_strategies_traitgym.sh` runs every pooling, classifier and ensemble on TraitGym (`SKIP_EMBED=1` reruns only the metrics).

## Acknowledgements and citation

The `bend` package, the baseline models and the BEND variant datasets come from BEND:

    @inproceedings{marin2024bend,
      title={{BEND}: Benchmarking {DNA} Language Models on Biologically Meaningful Tasks},
      author={Frederikke Isa Marin and Felix Teufel and Marc Horlacher and Dennis Madsen and Dennis Pultz and Ole Winther and Wouter Boomsma},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=uKB4cFNQFg}
    }

Please also cite the data and models you use: DeepSEA ([Zhou & Troyanskaya, 2015](https://www.nature.com/articles/nmeth.3547)), ClinVar ([Landrum et al., 2019](https://doi.org/10.1093/nar/gkz972)), [TraitGym](https://huggingface.co/datasets/songlab/TraitGym), [GENCODE](https://www.gencodegenes.org/), [DNABERT](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680), [DNABERT-2](https://arxiv.org/abs/2306.15006), [Nucleotide Transformer](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2), [HyenaDNA](https://arxiv.org/abs/2306.15794), [GROVER](https://www.nature.com/articles/s42256-024-00872-0), [Borzoi](https://doi.org/10.1038/s41588-024-02053-6) and [AlphaGenome](https://www.biorxiv.org/content/10.1101/2025.06.25.661532v1).

## License

See [LICENSE](LICENSE) (Creative Commons Attribution 4.0). The code in `bend/` is derived from [BEND](https://github.com/frederikkemarin/BEND); keep its original license notice when redistributing.
