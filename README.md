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
├── __init__.py                  # exposes bend.utils.embedders
├── models/                      # model architectures loaded by the embedders
│   ├── dnabert2.py              # DNABERT-2 (local copy, no flash attention)
│   ├── dnabert2_padding.py      # padding helpers for dnabert2.py
│   ├── gena_lm.py               # GENA-LM BERT
│   └── hyena_dna.py             # HyenaDNA + character tokenizer
└── utils/
    ├── embedders.py             # one wrapper class per model, common .embed() interface
    ├── retrieve_from_bed.py     # Annotation: bed file + FASTA -> DNA sequences
    └── download.py              # download helper for Zenodo-hosted checkpoints (GROVER)
scripts/
├── download_bend.py                       # download the BEND data share
├── predict_variant_effects.py             # zero-shot variant scoring (main script)
├── run_variant_effects.sh                 # runs every model on both BEND variant files
├── fine-tune_variant_effects.py           # LoRA fine-tuning + test AUROC/AUPRC (DNABERT-2)
├── fine-tune_variant_effects_crossval.py  # k-fold version of the fine-tuning (validation loss only)
└── get_genes.py                           # annotate expression variants with overlapping genes
```

## Setup

The project targets **Python 3.10** (pinned to 3.10.13 in `.python-version`; `setup.py` accepts up to 3.11.9).

```bash
git clone https://github.com/tmgoribeiro/BenchmarkVariantEffect.git
cd BenchmarkVariantEffect
pip install -r requirements.txt
pip install -e .          # installs the `bend` package in editable mode
```

`transformers` is pinned to `<=4.33.3` because the local DNABERT-2, GENA-LM and HyenaDNA code depends on that API. `peft` is pinned to `0.10.0` for the same reason: newer versions import `transformers.Cache`, which only exists from transformers 4.36.
`bend/utils/embedders.py` imports `alphagenome_pytorch` and `peft` at module level, so both must be installed even if you only use another model.

Some scripts and models need packages that are **not** in `requirements.txt`:

| Needed for | Install |
| --- | --- |
| Borzoi embedder | `pip install --no-deps "borzoi-pytorch>=0.5.0" intervaltree sortedcontainers` (without `--no-deps`, pip upgrades `transformers` and breaks GENA-LM) |
| `scripts/download_bend.py` | `pip install requests beautifulsoup4 lxml` |
| `scripts/get_genes.py` | `pip install pyranges` |
| GPN embedder | the `gpn` package |

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

`scripts/run_variant_effects.sh` runs every model on both BEND variant files. Run it from the repository root.

### Windows and embedding index

The window around each variant is built so that the variant is always at index `extra_context`:

- **Bidirectional models** get `[start − extra_context, start + extra_context)`, exactly `2 × extra_context` nucleotides.
- **Autoregressive models** (`hyenadna`) get left context only: `[start − extra_context, start + 1)`, so the variant is the last nucleotide. These models are causal, so the embedding of any position before the variant is identical for both alleles. An index that is off by one would give a distance of exactly 0 for every variant.

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
| `genalm` | `GENALMEmbedder` | `AIRI-Institute/gena-lm-bert-large-t2t` | BPE, upsampled |
| `grover` | `GROVEREmbedder` | `pretrained_models/grover` | BPE, upsampled |
| `hyenadna` | `HyenaDNAEmbedder` | `pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen` | autoregressive; needs Git LFS for the download |
| `dnabert` | `DNABertEmbedder` | `pretrained_models/dnabert/6-new-12w-0` | use `--kmer` |
| `gpn` | `GPNEmbedder` | `songlab/gpn-brassicales` | trained on plants, not human |

The embedders can also be used directly:

```python
from bend.utils.embedders import DNABert2Embedder

embedder = DNABert2Embedder('zhihan1996/DNABERT-2-117M')
embeddings = embedder.embed(['AGGATGCCGAGAGTATATGGGA'], upsample_embeddings=True)  # list of (1, length, dim) arrays
```

### Output

Each run writes to `runs/<model>/<run_name>/`:

| File | Content |
| --- | --- |
| `embeddings_<run_name>.pkl` | one pickled tuple `(index, embedding_ref, embedding_alt)` per variant, written one after another |
| `cosine_distance_<run_name>.csv` | the input bed plus a `distance` column; `start` and `end` hold the window coordinates |
| `log.txt` | one line appended per run: timestamp, run name, elapsed time, number of variants, AUROC (`AUC=`) and AUPRC |

The `.pkl` file holds many sequential pickles, so read it in a loop:

```python
import pickle
records = []
with open('runs/dnabert2/disease_dnabert2/embeddings_disease_dnabert2.pkl', 'rb') as f:
    while True:
        try:
            records.append(pickle.load(f))
        except EOFError:
            break
```

## Fine-tuning DNABERT-2 with LoRA (experimental)

```bash
python scripts/fine-tune_variant_effects.py <bed_file> dnabert2 zhihan1996/DNABERT-2-117M <genome_fasta> <batch_size> <epochs> \
    [--extra_context 256] [--max_variants N]
```

- `--max_variants` uses only the first N variants, for quick tests. All variants are used by default.
- The window and embedding index are built as in the zero-shot script.
- Only `DNABert2Embedder` implements the methods the fine-tuning needs (`prepare_for_finetuning` and `training_embed`). The other choices in the argument list fail.
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
- The `--out_dir` option of `fine-tune_variant_effects_crossval.py` is not used yet, so no checkpoints are saved.
- The automatic checkpoint downloads use Linux commands (`wget` for GROVER; `rm`, `mkdir -p` and `git lfs` for HyenaDNA). On other systems, download the checkpoints manually into `pretrained_models/`.

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
- [GENA-LM](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v1)
- [HyenaDNA](https://arxiv.org/abs/2306.15794)
- [GROVER](https://www.nature.com/articles/s42256-024-00872-0)
- [GPN](https://www.biorxiv.org/content/10.1101/2022.08.22.504706v2)
- [Borzoi](https://doi.org/10.1038/s41588-024-02053-6)
- [AlphaGenome](https://www.biorxiv.org/content/10.1101/2025.06.25.661532v1)

## License

See [LICENSE](LICENSE) (Creative Commons Attribution 4.0). The code in `bend/` is derived from [BEND](https://github.com/frederikkemarin/BEND); keep its original license notice when redistributing.
