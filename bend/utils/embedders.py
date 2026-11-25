'''
embedders.py
------------
Wrapper classes for embedding sequences with pretrained DNA language models using a common interface.
The wrapper classes handle loading the models and tokenizers, and embedding the sequences. As far as possible,
models are downloaded automatically.
They also handle removal of special tokens, and optionally upsample the embeddings to the original sequence length.

Embedders can be used as follows. Please check the individual classes for more details on the arguments.

``embedder = EmbedderClass(model_name, some_additional_config_argument=6)``

``embedding = embedder(sequence, remove_special_tokens=True, upsample_embeddings=True)``

'''
import math
import os
from functools import partial
from typing import Iterable, List

import numpy as np
import torch
from alphagenome_pytorch import AlphaGenome
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from transformers import (AutoModelForMaskedLM, AutoTokenizer, BertConfig, BertModel, BertTokenizer,
                          logging)

from bend.models.dnabert2 import BertForMaskedLM as DNABert2BertForMaskedLM
from bend.models.hyena_dna import CharacterTokenizer, HyenaDNAPreTrainedModel
from bend.utils.download import download_model_zenodo

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseEmbedder():
    """Base class for embedders.
    All embedders should inherit from this class.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the embedder. Calls `load_model` with the given arguments.

        Parameters
        ----------
        *args
            Positional arguments. Passed to `load_model`.
        **kwargs
            Keyword arguments. Passed to `load_model`.
        """
        self.load_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """Load the model. Should be implemented by the inheriting class."""
        raise NotImplementedError
    
    def embed(self, sequences:str, *args, **kwargs):
        """Embed a sequence. Should be implemented by the inheriting class.
        
        Parameters
        ----------
        sequences : str
            The sequences to embed.
        """
        raise NotImplementedError

    def __call__(self, sequence: str, *args, **kwargs):
        """Embed a single sequence. Calls `embed` with the given arguments.
        
        Parameters
        ----------
        sequence : str
            The sequence to embed.
        *args
            Positional arguments. Passed to `embed`.
        **kwargs
            Keyword arguments. Passed to `embed`.

        Returns
        -------
        np.ndarray
            The embedding of the sequence.
        """
        return self.embed([sequence], *args, disable_tqdm=True, **kwargs)[0]


##
## DNABert https://doi.org/10.1093/bioinformatics/btab083
## Download from https://github.com/jerryji1993/DNABERT

class DNABertEmbedder(BaseEmbedder):
    '''Embed using the DNABert model https://doi.org/10.1093/bioinformatics/btab083'''

    def load_model(self, 
                   model_path: str = '../../external-models/DNABERT/', 
                   kmer: int = 6, 
                   **kwargs):
        """Load the DNABert model.

        Parameters
        ----------
        model_path : str
            The path to the model directory. Defaults to "../../external-models/DNABERT/".
            The DNABERT models need to be downloaded manually as indicated in the DNABERT repository at https://github.com/jerryji1993/DNABERT.
        kmer : int
            The kmer size of the model. Defaults to 6.

        """

        if not os.path.exists(model_path):
            print(f'Path {model_path} does not exists, check if the wrong path was given. If not download from https://github.com/jerryji1993/DNABERT')

        config = BertConfig.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert_model = BertModel.from_pretrained(model_path, config=config)
        self.bert_model.to(device)
        self.bert_model.eval()

        self.kmer = kmer

    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = True, upsample_embeddings: bool = False):
        """
        Embed a list of sequences.

        Parameters
        ----------
        sequences : List[str]
            The sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the special tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to the length of the input sequence. Defaults to False.
        
        Returns
        -------
        List[np.ndarray]
            The embeddings of the sequences.
        """
        embeddings = []
        with torch.no_grad():
            for sequence in tqdm(sequences, disable=disable_tqdm):
                sequence = [sequence]
                kmers = self._seq2kmer_batch(sequence, self.kmer)
                model_input = self.tokenizer.batch_encode_plus(kmers, 
                                                               add_special_tokens=True,
                                                               return_tensors='pt', 
                                                               )["input_ids"]
                
                if model_input.shape[1] > 512:
                    model_input = torch.split(model_input, 512, dim=1)
                    output = []
                    for chunk in model_input: 
                        output.append(self.bert_model(chunk.to(device))[0].detach().cpu())
                    output = torch.cat(output, dim=1).numpy()
                else:
                    output = self.bert_model(model_input.to(device))[0].detach().cpu().numpy()
                embedding = output

                if upsample_embeddings:
                    embedding = self._repeat_embedding_vectors(embedding)

                embeddings.append(embedding[:,1:-1] if remove_special_tokens else embedding)

        return embeddings

    @staticmethod
    def _seq2kmer(seq, k):
        """
        Convert original sequence to kmers
        
        Arguments:
        seq -- str, original sequence.
        k -- int, kmer of length k specified.
        
        Returns:
        kmers -- str, kmers separated by space
        """
        kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
        kmers = " ".join(kmer)
        return kmers

    def _seq2kmer_batch(self, batch, k=3):
        return list(map(partial(self._seq2kmer, k = k), batch))
    
    # repeating.
    # GATTTATTAGGGGAGATTTTATATATCCCGA
    # kmer =3, input = 31 --> embedding = 29 --> repeat first and last once.
    # kmer =3, input = 32 --> embedding = 30 --> repeat first and last once.

    # kmer=4 input = 31 --> embedding = 28 --> repeat first once and last twice.
    # kmer=4 input = 32 --> embedding = 29
    # kmer=4 input = 33 --> embedding = 30

    # kmer=5 input = 31 --> embedding = 27 --> repeat first twice and last twice.
    # kmer=5 input = 32 --> embedding = 28 --> repeat first twice and last twice.

    # kmer=6 input = 31 --> embedding = 26 --> repeat first twice and last three times.
    def _repeat_embedding_vectors(self, embeddings: np.ndarray, has_special_tokens: bool = True):
        '''Repeat embeddings at sequence edges to match input length'''
        if has_special_tokens:
            cls_vector = embeddings[:, [0]]
            sep_vector = embeddings[:, [-1]]
            embeddings = embeddings[:,1:-1]

        # repeat first and last embedding
        if self.kmer == 3:
            embeddings = np.concatenate([embeddings[:, [0]], embeddings, embeddings[:, [-1]]], axis=1)
        elif self.kmer == 4:
            embeddings = np.concatenate([embeddings[:, [0]], embeddings, embeddings[:, [-1]], embeddings[:, [-1]]], axis=1)
        elif self.kmer == 5:
            embeddings = np.concatenate([embeddings[:, [0]], embeddings, embeddings[:, [0]], embeddings[:, [-1]], embeddings[:, [-1]]], axis=1)
        elif self.kmer == 6:
            embeddings = np.concatenate([embeddings[:, [0]], embeddings, embeddings[:, [0]], embeddings[:, [-1]], embeddings[:, [-1]], embeddings[:, [-1]]], axis=1)
        
        if has_special_tokens:
            embeddings = np.concatenate([cls_vector, embeddings, sep_vector], axis=1)

        return embeddings





# https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
class NucleotideTransformerEmbedder(BaseEmbedder):
    """
    Embed using the Nuclieotide Transformer (NT) model https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
    """

    def load_model(self, model_name, **kwargs):
        """
        Load the Nuclieotide Transformer (NT) model.

        Parameters
        ----------
        model_name : str
            The name of the model to load.
            When providing a name, the model will be loaded from the HuggingFace model hub.
            Alternatively, you can provide a path to a local model directory. We check whether the model_name
            contains 'v2' or 'v3' to determine whether we need to follow the V2 model API or not.
        """
        if 'v2' in model_name or 'v3' in model_name:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.max_seq_len = 12282 # "model_max_length": 2048, --> 12,288
            self.max_tokens = 2048
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.max_seq_len = 5994 # "model_max_length": 1000, 6-mer --> 6000
            self.max_tokens = 1000
        self.model.to(device)
        self.model.eval()

    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = True, upsample_embeddings: bool = False):
        """
        Embed sequences using the Nuclieotide Transformer (NT) model.
        
        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
             Whether to remove the special tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to the length of the input sequence. Defaults to False.

        Returns
        -------
        List[np.ndarray]
            List of embeddings.
        """
        embeddings = []
        with torch.no_grad():
            for s in tqdm(sequences, disable=disable_tqdm):
                s_chunks = [s[chunk : chunk + self.max_seq_len] for chunk in  range(0, len(s), self.max_seq_len)] # split into chunks
                embedded_seq = []
                for chunk in s_chunks: # embed each chunk
                    tokens_ids = self.tokenizer(chunk, return_tensors = 'pt')['input_ids'].int().to(device)
                    # inputs longer than max_tokens do not fit into the model and are embedded in pieces
                    outs = [self.model(item, output_hidden_states=True)['hidden_states'][-1].detach().cpu().numpy()
                            for item in torch.split(tokens_ids, self.max_tokens, dim=-1)]
                    outs = np.concatenate(outs, axis=1)

                    if upsample_embeddings:
                        outs = self._repeat_embedding_vectors(self.tokenizer.convert_ids_to_tokens(tokens_ids[0]), outs)

                    embedded_seq.append(outs[:,1:] if remove_special_tokens else outs)

                embeddings.append(np.concatenate(embedded_seq, axis=1))

        return embeddings
    
    @staticmethod
    def _repeat_embedding_vectors(tokens: Iterable[str], embeddings: np.ndarray, has_special_tokens: bool = True):
        '''
        Nucleotide transformer uses 6-mer embedding, but single-embedding for remaining nucleotides.
        '''
        assert len(tokens) == embeddings.shape[1], 'Number of tokens and embeddings must match.'
        new_embeddings = []
        for idx, token in enumerate(tokens):

            if has_special_tokens and idx == 0:
                new_embeddings.append(embeddings[:, [idx]]) # (1, hidden_dim)
                continue
            token_embedding = embeddings[:, [idx]] # (1, hidden_dim)
            new_embeddings.extend([token_embedding] * len(token))

        # list of (1,1, 768) arrays
        new_embeddings = np.concatenate(new_embeddings, axis=1)
        return new_embeddings


class HyenaDNAEmbedder(BaseEmbedder):
    '''Embed using the HyenaDNA model https://arxiv.org/abs/2306.15794'''
    def load_model(self, model_path = 'pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen', **kwargs):
        """
        Load the HyenaDNA model.

        Parameters
        ----------
        model_path : str, optional
            Path to the model checkpoint. Defaults to 'pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen'.
            If the path does not exist, the model will be downloaded from HuggingFace. Rather than just downloading the model,
            HyenaDNA's `from_pretrained` method relies on cloning the HuggingFace-hosted repository, and using git lfs to download the model.
            This requires git lfs to be installed on your system, and will fail if it is not.
        """
        checkpoint_path, model_name = os.path.split(model_path)
        max_lengths = {
            'hyenadna-tiny-1k-seqlen': 1024,
            'hyenadna-small-32k-seqlen': 32768,
            'hyenadna-medium-160k-seqlen': 160000,
            'hyenadna-medium-450k-seqlen': 450000,
            'hyenadna-large-1m-seqlen': 1_000_000,
        }
        self.max_length = max_lengths[model_name]

        is_git_lfs_repo = os.path.exists('.git/hooks/pre-push')
        model = HyenaDNAPreTrainedModel.from_pretrained(
            checkpoint_path,
            model_name,
            download=not os.path.exists(model_path),
            device=device,
        )
        model.eval()
        model.to(device)
        self.model = model

        # NOTE the git lfs download command will add this,
        # but we actually dont use LFS for BEND itself.
        if not is_git_lfs_repo:
            try:
                os.remove('.git/hooks/pre-push')
            except FileNotFoundError:
                pass

        # create tokenizer - NOTE this adds CLS and SEP tokens when add_special_tokens=False
        self.tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=self.max_length + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side='left', # since HyenaDNA is causal, we pad on the left
        )

    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = True, upsample_embeddings: bool = False):
        '''Embeds a list of sequences using the HyenaDNA model.
        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the CLS and SEP tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Only provided for compatibility with other embedders. HyenaDNA embeddings are already the same length as the input sequence.
        Returns
        -------
        embeddings : List[np.ndarray]
            List of embeddings.
        '''
        embeddings = []
        with torch.inference_mode():
            for s in tqdm(sequences, disable=disable_tqdm):
                chunks = [s[chunk : chunk + self.max_length] for chunk in  range(0, len(s), self.max_length)] # split into chunks
                embedded_chunks = []
                for chunk in chunks:
                    tok_seq = self.tokenizer(chunk)["input_ids"] # adds CLS and SEP tokens (0=CLS, 1=EOS)
                    tok_seq = torch.LongTensor(tok_seq).unsqueeze(0).to(device) # unsqueeze for batch dim

                    output = self.model(tok_seq)
                    if remove_special_tokens:
                        output = output[:,1:-1]

                    embedded_chunks.append(output.detach().cpu().numpy())

                embeddings.append(np.concatenate(embedded_chunks, axis=1))

        return embeddings



class DNABert2Embedder(BaseEmbedder):
    """
    Embed using the DNABERT2 model https://arxiv.org/pdf/2306.15006.pdf
    """
    def load_model(self, model_name = "zhihan1996/DNABERT-2-117M", **kwargs):
        """
        Load the DNABERT2 model.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to load. Defaults to "zhihan1996/DNABERT-2-117M".
            When providing a name, the model will be loaded from the HuggingFace model hub.
            Alternatively, you can provide a path to a local model directory.
        """
        # keep the source in this repo to avoid using flash attn.
        self.model = DNABert2BertForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(device)

        # https://github.com/Zhihan1996/DNABERT_2/issues/2
        self.max_length = 10000 #nucleotides.

    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = True, upsample_embeddings: bool = True):
        '''Embeds a list sequences using the DNABERT2 model.
        
        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the CLS and SEP tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to match the length of the input sequences. Defaults to False.

        Returns
        -------
        embeddings : List[np.ndarray]
            List of embeddings.
        '''
        embeddings = []
        with torch.no_grad():
            for sequence in tqdm(sequences, disable=disable_tqdm):

                chunks = [sequence[chunk : chunk + self.max_length] for chunk in  range(0, len(sequence), self.max_length)] # split into chunks

                embedded_chunks = []
                for n_chunk, chunk in enumerate(chunks):

                    input_ids = self.tokenizer(chunk, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
                    output = self.model(input_ids.to(device), output_hidden_states=True)['hidden_states'][-1].detach().cpu().numpy()
                    if upsample_embeddings:
                        output = self._repeat_embedding_vectors(self.tokenizer.convert_ids_to_tokens(input_ids[0]), output)

                    # for intermediate chunks the special tokens need to go.
                    # if we only have 1 chunk, keep them for now.
                    if len(chunks) != 1:
                        if n_chunk == 0:
                            output = output[:,:-1] # no SEP
                        elif n_chunk == len(chunks) - 1:
                            output = output[:,1:] # no CLS
                        else:
                            output = output[:,1:-1] # no CLS and no SEP

                    embedded_chunks.append(output)

                embedding = np.concatenate(embedded_chunks, axis=1)

                if remove_special_tokens:
                    embedding = embedding[:,1:-1]

                embeddings.append(embedding)

        return embeddings

    def prepare_for_finetuning(self, r=8, lora_alpha=16, target_modules=None, lora_dropout=0.1):

        """Wrap self.model with a LoRA adapter and put it in train mode."""
        if target_modules is None:
            # confirm these against self.model.named_modules() first
            target_modules = ['Wqkv', 'attention.output.dense']

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()
        self.model.print_trainable_parameters()

    def training_embed(self, sequences: List[str], position_idx=0, upsample_embeddings=True, remove_special_tokens=True, use_grads=True):

        context = torch.enable_grad() if use_grads else torch.no_grad()

        with context:
            encoded = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
                return_token_type_ids=False,
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = output["hidden_states"]  # (batch, seq_len, hidden_dim)

            pooled = []
            for i in range(input_ids.shape[0]):
                seq_len = attention_mask[i].sum().item()  # drop padding for this sequence
                seq_ids = input_ids[i, :seq_len]
                seq_hidden = hidden[i, :seq_len]  # (seq_len_tokens, dim) — still tracked by autograd

                if upsample_embeddings:
                    tokens = self.tokenizer.convert_ids_to_tokens(seq_ids)
                    seq_hidden = self._repeat_embedding_vectors_tensor(tokens, seq_hidden)  # (seq_len_nt, dim)

                if remove_special_tokens:
                    seq_hidden = seq_hidden[1:-1]  # CLS/SEP occupy single positions even after upsampling, matching embed()'s order

                pooled.append(seq_hidden[position_idx])

            return torch.stack(pooled)

    # GATTTATTAGGGGAGATTTTATATATCCCGA
    # ['[CLS]', 'G', 'ATTTATT', 'AGGGG', 'AGATT', 'TTATAT', 'ATCCCG', 'A', '[SEP]']
    @staticmethod
    def _repeat_embedding_vectors(tokens: Iterable[str], embeddings: np.ndarray, has_special_tokens: bool = True):
        '''
        Byte-pair encoding merges a variable number of letters into one token.
        We need to repeat each token's embedding vector for each letter in the token.
        '''
        assert len(tokens) == embeddings.shape[0], 'Number of tokens and embeddings must match.'
        new_embeddings = []
        for idx, token in enumerate(tokens):

            slice_i = embeddings[idx : idx + 1, :]

            if has_special_tokens and (idx == 0 or idx == len(tokens) - 1):
                new_embeddings.append(slice_i)
                continue
            if token == "[UNK]":
                new_embeddings.append(slice_i)
            else:
                new_embeddings.extend([slice_i] * len(token))

        # list of (1,1, 768) arrays
        new_embeddings = np.concatenate(new_embeddings, axis=0)
        return new_embeddings[None, :, :]

    @staticmethod
    def _repeat_embedding_vectors_tensor(tokens: Iterable[str], embeddings: torch.Tensor, has_special_tokens: bool = True):
        '''
        Tensor-native version of _repeat_embedding_vectors — preserves the autograd graph.
        embeddings: (num_tokens, dim), no batch dim.
        '''
        assert len(tokens) == embeddings.shape[0], 'Number of tokens and embeddings must match.'
        new_embeddings = []
        for idx, token in enumerate(tokens):
            slice_i = embeddings[idx:idx + 1, :]  # (1, dim)

            if has_special_tokens and (idx == 0 or idx == len(tokens) - 1):
                new_embeddings.append(slice_i)
                continue
            if token == "[UNK]":
                new_embeddings.append(slice_i)
            else:
                new_embeddings.append(slice_i.expand(len(token), -1))  # view-based repeat, autograd-safe

        return torch.cat(new_embeddings, dim=0)  # (seq_len_nucleotides, dim)


class GROVEREmbedder(BaseEmbedder):
    '''Embed using the GROVER model https://www.biorxiv.org/content/10.1101/2023.07.19.549677v2'''

    def load_model(self, model_path: str = "pretrained_models/grover" , **kwargs):
        """Load the GROVER model.

        Parameters
        ----------
        model_path : str
            The path to the model directory.
            If the model path does not exist, it will be downloaded from https://zenodo.org/records/8373117
        """
        # download model if not exists
        if not os.path.exists(model_path):
            print(f'Path {model_path} does not exists, model is downloaded from https://zenodo.org/records/8373117')
            download_model_zenodo(
                base_url = 'https://zenodo.org/records/8373117',
                destination_dir = model_path
                )


        self.model = BertModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)

        self.model.to(device)
        self.model.eval()

        self.max_length = 510 # NOTE this is BPE tokens, not bp.

        self.max_token_length = max([len(token) for token in self.tokenizer.vocab.keys()])


    def max_match_tokenize(self, sequence: str) -> List[str]:
        """
        Tokenize a sequence using max match. 
        We have to do this as we do not have access to the BPE tokenizer used by GROVER.
        We only have access to the vocabulary, so we find a sequence-to-token assignment
        that uses the longest possible tokens.

        Parameters
        ----------
        sequence : str
            The sequence to tokenize.

        Returns
        -------
        List[str]
            The tokenized sequence.
        """
        tokens = []
        i = 0
        while i < len(sequence):
            max_token = None
            for j in range(i + self.max_token_length, i, -1):
                candidate = sequence[i:j]
                if candidate in self.tokenizer.vocab:
                    max_token = candidate
                    break
            if max_token is None:
                # a character that is in no token becomes an unknown token
                tokens.append(self.tokenizer.unk_token)
                i += 1
            else:
                tokens.append(max_token)
                i += len(max_token)
        return tokens


    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = True, upsample_embeddings: bool = False):
        '''Embeds a list sequences using the GROVER model.
        Note that the BPE tokenizer that GROVER used is not provided, we only
        have access to the vocabulary used for tokenization. Instead,
        we use max match to tokenize the sequence, so that each subsequence gets
        tokenized as its longest token in the vocabulary. Not certain that this is 
        identical to what a correctly instantiated BPE tokenizer would do.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Whether to remove the CLS and SEP tokens from the embeddings. Defaults to True.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to match the length of the input sequences. Defaults to False.

        Returns
        -------
        embeddings : List[np.ndarray]
            List of embeddings.
        '''
        embeddings = []
        with torch.no_grad():
            for sequence in tqdm(sequences, disable=disable_tqdm):

                # pre-tokenize to BPE words
                sequence_toks = self.max_match_tokenize(sequence)
                chunks = [sequence_toks[chunk : chunk + self.max_length] for chunk in  range(0, len(sequence_toks), self.max_length)] # split bpe tokens into chunks
                embedded_chunks = []
                for n_chunk, chunk in enumerate(chunks):

                    input_ids = self.tokenizer(' '.join(chunk), return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
                    output = self.model(input_ids.to(device))[0].detach().cpu().numpy()

                    if upsample_embeddings:
                        output = self._repeat_embedding_vectors(self.tokenizer.convert_ids_to_tokens(input_ids[0]), output)

                    # for intermediate chunks the special tokens need to go.
                    # if we only have 1 chunk, keep them for now.
                    if len(chunks) != 1:
                        if n_chunk == 0:
                            output = output[:,:-1] # no SEP
                        elif n_chunk == len(chunks) - 1:
                            output = output[:,1:] # no CLS
                        else:
                            output = output[:,1:-1] # no CLS and no SEP

                    embedded_chunks.append(output)

                embedding = np.concatenate(embedded_chunks, axis=1)

                if remove_special_tokens:
                    embedding = embedding[:,1:-1]

                if upsample_embeddings and remove_special_tokens:
                    assert len(sequence) == embedding.shape[1], f'Number of tokens and embeddings must match. {len(sequence)} != {embedding.shape[1]}'
                elif upsample_embeddings:
                    assert len(sequence)+ 2 == embedding.shape[1], f'Number of tokens and embeddings must match. {len(sequence)+ 2} != {embedding.shape[1]}'

                embeddings.append(embedding)

        return embeddings
    
    # GATTTATTAGGGGAGATTTTATATATCCCGA
    # ['[CLS]', 'G', 'ATTTATT', 'AGGGG', 'AGATT', 'TTATAT', 'ATCCCG', 'A', '[SEP]']
    @staticmethod
    def _repeat_embedding_vectors(tokens: Iterable[str], embeddings: np.ndarray, has_special_tokens: bool = True):
        '''
        Byte-pair encoding merges a variable number of letters into one token.
        We need to repeat each token's embedding vector for each letter in the token.
        '''
        assert len(tokens) == embeddings.shape[1], 'Number of tokens and embeddings must match.'
        new_embeddings = []
        for idx, token in enumerate(tokens):

            if has_special_tokens and (idx == 0 or idx == len(tokens) - 1):
                new_embeddings.append(embeddings[:, [idx]]) # (1, 768)
                continue
            token_embedding = embeddings[:, [idx]] # (1, 768)
            if token == '[UNK]':
                new_embeddings.extend([token_embedding])
            else:
                new_embeddings.extend([token_embedding] * len(token))

        # list of (1,1, 768) arrays
        new_embeddings = np.concatenate(new_embeddings, axis=1)
        return new_embeddings


# Class for one-hot encoding.
categories_4_letters_unknown = ['A', 'C', 'G', 'N', 'T']

class EncodeSequence:
    def __init__(self, nucleotide_categories = categories_4_letters_unknown):
        
        self.nucleotide_categories = nucleotide_categories
        
        self.label_encoder = LabelEncoder().fit(self.nucleotide_categories)
        
    
    def transform_integer(self, sequence, return_onehot = False): # integer/onehot encode sequence
        if isinstance(sequence, np.ndarray):
            return sequence
        if isinstance(sequence[0], str):  # if input is str 
            sequence = np.array(list(sequence))
        
        sequence = self.label_encoder.transform(sequence)
        
        if return_onehot:
            sequence = np.eye(len(self.nucleotide_categories))[sequence]
        return sequence


class AlphaGenomeEmbedder(BaseEmbedder):
    '''Embed using the AlphaGenome model (alphagenome-pytorch). Needs a CUDA GPU.'''

    def load_model(self, model_path = 'pretrained_models/alphagenome/model_all_folds.safetensors', **kwargs):
        """
        Load the AlphaGenome model.

        Parameters
        ----------
        model_path : str, optional
            Path to the safetensors checkpoint. Defaults to 'pretrained_models/alphagenome/model_all_folds.safetensors'.
        """
        self.model = AlphaGenome.from_pretrained(model_path, device='cuda')
        self.model.eval()
        self.encoder = EncodeSequence(nucleotide_categories=['A', 'C', 'G', 'T'])

    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = False, upsample_embeddings: bool = False):
        '''Embeds a list of sequences using the AlphaGenome model, one 1 bp-resolution embedding per nucleotide (human head).

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed. Their length must be a multiple of 2048.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens, upsample_embeddings : bool, optional
            Only provided for compatibility with other embedders. AlphaGenome has no special tokens and one embedding per nucleotide.

        Returns
        -------
        embeddings : List[np.ndarray]
            List of embeddings.
        '''
        embeddings = []
        with torch.inference_mode():
            for s in tqdm(sequences, disable=disable_tqdm):
                dna_onehot = self.encoder.transform_integer(sequence=s, return_onehot=True)
                dna_onehot = torch.tensor(dna_onehot).unsqueeze(0).cuda()

                emb = self.model.encode(dna_onehot, organism_index=0, resolutions=(1,))
                embeddings.append(emb['embeddings_1bp'].cpu().numpy())

        return embeddings


class BorzoiEmbedder(BaseEmbedder):
    '''Embed using the Borzoi model https://doi.org/10.1038/s41588-024-02053-6'''

    def load_model(self, model_name: str = 'johahi/borzoi-replicate-0', **kwargs):
        """
        Load the Borzoi model.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to load. Defaults to "johahi/borzoi-replicate-0".
            When providing a name, the model will be loaded from the HuggingFace model hub.
            Alternatively, you can provide a path to a local model directory.

        Raises
        ------
        ModuleNotFoundError
            If the borzoi_pytorch module is not installed.
        ImportError
            If the installed borzoi_pytorch module is older than 0.5.0.

        Notes
        -----
        The borzoi_pytorch module can be installed with `pip install --no-deps "borzoi-pytorch>=0.5.0" intervaltree sortedcontainers`.
        Without --no-deps, pip upgrades transformers past the version pinned in requirements.txt.
        """
        try:
            from borzoi_pytorch import Borzoi
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Borzoi requires borzoi-pytorch. Install with: pip install --no-deps "borzoi-pytorch>=0.5.0" intervaltree sortedcontainers')

        # Before 0.5.0 the relative positions of the attention layers are fixed to the 524 kb training input,
        # which silently gives wrong attention for shorter sequences.
        from importlib import metadata
        from packaging import version
        if version.parse(metadata.version('borzoi-pytorch')) < version.parse('0.5.0'):
            raise ImportError('Borzoi requires borzoi-pytorch>=0.5.0. Install with: pip install --no-deps "borzoi-pytorch>=0.5.0" intervaltree sortedcontainers')

        self.model = Borzoi.from_pretrained(model_name)
        # The crop keeps the central 6144 bins of the 524 kb training input and raises an error for shorter sequences.
        # A target length of -1 disables it, so that we get one embedding for every bin of the input.
        self.model.crop.target_length = -1
        self.model.to(device)
        self.model.eval()

        self.max_length = 524288 # nucleotides, the input length Borzoi was trained on.
        self.bin_size = 32 # nucleotides represented by each embedding vector.
        self.length_multiple = 128 # the U-Net pools 7 times, so the input length has to be a multiple of 2**7.
        self.encoder = EncodeSequence() # A, C, G, N, T

    def embed(self, sequences: List[str], disable_tqdm: bool = False, remove_special_tokens: bool = True, upsample_embeddings: bool = False):
        """
        Embed sequences using the Borzoi model.

        Parameters
        ----------
        sequences : List[str]
            List of sequences to embed.
        disable_tqdm : bool, optional
            Whether to disable the tqdm progress bar. Defaults to False.
        remove_special_tokens : bool, optional
            Only provided for compatibility with other embedders. Borzoi does not use special tokens.
        upsample_embeddings : bool, optional
            Whether to upsample the embeddings to the length of the input sequence. Defaults to False.
            Borzoi returns one embedding vector per 32 bp bin, so upsampling repeats each vector 32 times.

        Returns
        -------
        List[np.ndarray]
            List of embeddings of shape (1, length, 1920), taken from the layer right before the output heads.
            The length is len(sequence) if upsample_embeddings is True, and ceil(len(sequence) / 32) otherwise.
        """
        embeddings = []
        with torch.inference_mode():
            for s in tqdm(sequences, disable=disable_tqdm):
                chunks = [s[chunk : chunk + self.max_length] for chunk in range(0, len(s), self.max_length)] # split into chunks
                embedded_chunks = []
                for chunk in chunks:
                    # (1, 4, length) in A, C, G, T order. Dropping the N column encodes N as all zeros.
                    dna_onehot = self.encoder.transform_integer(sequence=chunk.upper(), return_onehot=True)[:, [0, 1, 2, 4]]
                    dna_onehot = torch.tensor(dna_onehot, dtype=torch.float32).T.unsqueeze(0).to(device)

                    # pad with Ns up to the next valid length. The bins that only cover padding are removed below.
                    dna_onehot = torch.nn.functional.pad(dna_onehot, (0, -len(chunk) % self.length_multiple))

                    # The attention layers cache the relative positions of the first input they see and
                    # assume that length from then on, so they have to be recomputed for each input.
                    for module in self.model.modules():
                        if hasattr(module, 'positions'):
                            module.positions = None

                    output = self.model.final_joined_convs(self.model.get_embs_after_crop(dna_onehot))
                    output = output.permute(0, 2, 1).cpu().numpy() # (1, bins, 1920)
                    output = output[:, :math.ceil(len(chunk) / self.bin_size)]

                    if upsample_embeddings:
                        output = np.repeat(output, self.bin_size, axis=1)[:, :len(chunk)]

                    embedded_chunks.append(output)

                embedding = np.concatenate(embedded_chunks, axis=1)
                embeddings.append(embedding)

        return embeddings
