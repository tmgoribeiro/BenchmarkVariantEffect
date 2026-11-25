'''
Annotate the expression variants with the GENCODE genes they fall in (one row per variant-gene pair).
'''
import pandas as pd
import pyranges as pr

cols = ["chromosome", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
genes_raw = pd.read_csv(
    "./data/genomes/gencode.v50.genes_only.gtf",
    sep="\t", comment="#", names=cols,
    dtype={"chromosome": str}
)

genes_raw["gene_id"] = genes_raw["attribute"].str.extract(r'gene_id "([^"]+)"')
genes_raw["gene_name"] = genes_raw["attribute"].str.extract(r'gene_name "([^"]+)"')
genes_raw = genes_raw.rename(columns={"chromosome": "Chromosome", "start": "Start", "end": "End"})

genes = pr.PyRanges(genes_raw[["Chromosome", "Start", "End", "strand", "gene_id", "gene_name"]])

df = pd.read_csv("./data/variant_effects/variant_effects_expression.bed", sep="\t")
df["variant_idx"] = range(len(df))

query = pr.PyRanges(chromosomes=df["chromosome"], starts=df["start"], ends=df["end"] + 1)
query.variant_idx = df["variant_idx"].values

result_df = genes.join(query).df[["variant_idx", "gene_id", "gene_name"]]

df_annotated = df.merge(result_df, on="variant_idx", how="left")
n_intergenic = df_annotated["gene_name"].isna().sum()
df_annotated = df_annotated.drop(columns=["variant_idx"]).dropna(subset=["gene_name"])

df_annotated.to_csv("./data/variant_effects/variant_effects_expression_annotated.bed", sep="\t", index=False)

print(df_annotated.head(20))
print(f"\n{len(df)} variants -> {len(df_annotated)} rows after annotation")
print(f"{n_intergenic} variants had no gene overlap (intergenic) and were dropped")

print("\nTop 15 genes by variant count:")
print(df_annotated["gene_name"].value_counts().head(15))
