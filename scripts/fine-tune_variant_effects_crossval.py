'''
K-fold version of fine-tune_variant_effects.py: reports the mean and std of the best validation loss.
'''
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from bend.utils import Annotation, embedders


class VariantPairDataset(Dataset):
    def __init__(self, wt_sequences, alt_sequences, labels):
        assert len(wt_sequences) == len(alt_sequences) == len(labels)
        self.wt = wt_sequences
        self.alt = alt_sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "wt": self.wt[idx],
            "alt": self.alt[idx],
            "label": self.labels[idx],
        }


def contrastive_loss(emb_wt, emb_alt, labels, margin=10.0):
    labels = labels.float()
    cos_sim = F.cosine_similarity(emb_wt, emb_alt, dim=-1)
    distance = 1.0 - cos_sim  # cosine distance, range [0, 2]

    pathogenic_term = labels * torch.clamp(margin - distance, min=0.0) ** 2
    benign_term = (1.0 - labels) * distance ** 2

    return (pathogenic_term + benign_term).mean()


def run_epoch(embedder, loader, device, embedding_idx, optimizer=None):
    """One pass over `loader`. Trains if optimizer is given, else evaluates."""
    train_mode = optimizer is not None
    embedder.model.train() if train_mode else embedder.model.eval()

    total_loss = 0.0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            labels_batch = batch["label"].to(device)

            emb_wt = embedder.training_embed(batch["wt"], embedding_idx, use_grads=train_mode).to(device)
            emb_alt = embedder.training_embed(batch["alt"], embedding_idx, use_grads=train_mode).to(device)

            loss = contrastive_loss(emb_wt, emb_alt, labels_batch)

            if train_mode:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(embedder, train_loader, val_loader, device, args):
    """Trains for args.epochs and returns the best validation loss."""
    optimizer = torch.optim.AdamW([p for p in embedder.model.parameters() if p.requires_grad], lr=1e-5)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = run_epoch(embedder, train_loader, device, args.embedding_idx, optimizer)
        val_loss = run_epoch(embedder, val_loader, device, args.embedding_idx)
        print(f"  Epoch {epoch + 1}/{args.epochs}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")
        best_val_loss = min(best_val_loss, val_loss)

    return best_val_loss


def main():
    parser = argparse.ArgumentParser('Cross-validated fine-tuning of DNABERT-2 on variant pairs')
    parser.add_argument('bed_file', type=str, help='Path to the bed file')
    parser.add_argument('model', choices=['dnabert2'], type=str, help='Model to fine-tune (only DNABERT-2 supports it)')
    parser.add_argument('checkpoint', type=str, help='Path to or name of the model checkpoint')
    parser.add_argument('genome', type=str, help='Path to the reference genome fasta file')
    parser.add_argument('batch_size', type=int, help='Fine-tuning batch size')
    parser.add_argument('epochs', type=int, help='Fine-tuning epochs per fold')
    parser.add_argument('--extra_context', type=int, default=256, help='Number of extra nucleotides to include on each side of the sequence')
    parser.add_argument('--embedding_idx', type=int, default=None, help='Index of the embedding to use for computing the distance. Defaults to the position of the variant in the window')
    parser.add_argument('--max_variants', type=int, default=None, help='Only use the first N variants (for quick tests). Defaults to all')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of cross-validation folds')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    genome_annotation = Annotation(args.bed_file, reference_genome=args.genome)
    if args.max_variants is not None:
        genome_annotation.annotation = genome_annotation.annotation.iloc[:args.max_variants].reset_index(drop=True)

    # put the variant (0-based `start`) at index extra_context of every window
    genome_annotation.annotation['end'] = genome_annotation.annotation['start']
    if args.extra_context > 0:
        genome_annotation.extend_segments(extra_context_left=args.extra_context, extra_context_right=args.extra_context)

    variant_idx = args.extra_context if args.extra_context > 0 else 0
    if args.embedding_idx is None:
        args.embedding_idx = variant_idx
    genome_annotation.check_reference(variant_idx)

    wt_sequences, alt_sequences, labels = [], [], []
    for index, row in tqdm(genome_annotation.annotation.iterrows(), desc="Building sequence pairs"):
        dna = genome_annotation.get_dna_segment(index=index).upper().replace('N', 'A')
        dna_alt = list(dna)
        dna_alt[variant_idx] = row['alt']
        dna_alt = ''.join(dna_alt)

        wt_sequences.append(dna)
        alt_sequences.append(dna_alt)
        labels.append(row['label'])

    def loader(idx, shuffle):
        dataset = VariantPairDataset([wt_sequences[i] for i in idx], [alt_sequences[i] for i in idx], [labels[i] for i in idx])
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_val_losses = []
    for fold, (tr, val) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")

        # a fresh model for every fold
        embedder = embedders.DNABert2Embedder(args.checkpoint)
        embedder.prepare_for_finetuning()

        best_val_loss = train_model(embedder, loader(tr, shuffle=True), loader(val, shuffle=False), device, args)
        fold_val_losses.append(best_val_loss)
        print(f"Fold {fold + 1} best val loss: {best_val_loss:.4f}")

    fold_val_losses = np.array(fold_val_losses)
    print(f"\nCV val loss: {fold_val_losses.mean():.4f} +/- {fold_val_losses.std():.4f}")


if __name__ == '__main__':
    main()
