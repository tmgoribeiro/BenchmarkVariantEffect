'''
LoRA fine-tuning of DNABERT-2 with a contrastive loss on ref/alt pairs, then the test AUROC and AUPRC of the cosine distance.
'''
import argparse

import torch
import torch.nn.functional as F
from scipy import spatial
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
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


def main():

    parser = argparse.ArgumentParser('Fine-tune DNABERT-2 on variant pairs')
    parser.add_argument('bed_file', type=str, help='Path to the bed file')
    parser.add_argument('model', choices=['dnabert2'], type=str, help='Model to fine-tune (only DNABERT-2 supports it)')
    parser.add_argument('checkpoint', type=str, help='Path to or name of the model checkpoint')
    parser.add_argument('genome', type=str, help='Path to the reference genome fasta file')
    parser.add_argument('batch_size', type=int, help='Fine-tuning batch size')
    parser.add_argument('epochs', type=int, help='Fine-tuning epochs')
    parser.add_argument('--extra_context', type=int, default=256, help='Number of extra nucleotides to include on each side of the sequence')
    parser.add_argument('--embedding_idx', type=int, default=None, help = 'Index of the embedding to use for computing the distance. Defaults to the position of the variant in the window')
    parser.add_argument('--max_variants', type=int, default=None, help = 'Only use the first N variants (for quick tests). Defaults to all')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder = embedders.DNABert2Embedder(args.checkpoint)
    kwargs = {'disable_tqdm': True, 'upsample_embeddings': True}  # one embedding per nucleotide

    # load the bed file
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

    # 64 / 16 / 20 train / validation / test split, stratified by label
    trainval_idx, test_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.2, random_state=42,
                                          stratify=[labels[i] for i in trainval_idx])

    def loader(idx, shuffle):
        dataset = VariantPairDataset([wt_sequences[i] for i in idx], [alt_sequences[i] for i in idx], [labels[i] for i in idx])
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = loader(train_idx, shuffle=True)
    val_loader = loader(val_idx, shuffle=False)

    embedder.prepare_for_finetuning()
    optimizer = torch.optim.AdamW([p for p in embedder.model.parameters() if p.requires_grad], lr=1e-5)

    for epoch in range(args.epochs):
        embedder.model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            labels_batch = batch["label"].to(device)

            emb_wt = embedder.training_embed(batch["wt"], args.embedding_idx, use_grads=True).to(device)
            emb_alt = embedder.training_embed(batch["alt"], args.embedding_idx, use_grads=True).to(device)

            loss = contrastive_loss(emb_wt, emb_alt, labels_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        embedder.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch+1}/{args.epochs}"):
                labels_batch = batch["label"].to(device)

                emb_wt = embedder.training_embed(batch["wt"], args.embedding_idx, use_grads=False).to(device)
                emb_alt = embedder.training_embed(batch["alt"], args.embedding_idx, use_grads=False).to(device)

                val_loss += contrastive_loss(emb_wt, emb_alt, labels_batch).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: train loss = {epoch_loss / len(train_loader):.4f}, val loss = {val_loss:.4f}")

    # test: zero-shot score (cosine distance) of the fine-tuned model
    test_labels = [labels[i] for i in test_idx]
    test_distance = []
    for i in tqdm(test_idx, desc="Testing"):
        embedding_wt, embedding_alt = embedder.embed([wt_sequences[i], alt_sequences[i]], **kwargs)
        test_distance.append(spatial.distance.cosine(embedding_alt[0, args.embedding_idx], embedding_wt[0, args.embedding_idx]))

    print(f"Final Score: {roc_auc_score(test_labels, test_distance)}")
    print(f"AUPRC: {average_precision_score(test_labels, test_distance)}")


if __name__ == '__main__':
    main()
