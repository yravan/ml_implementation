"""
BERT Pretraining Datasets: MLM-only, NSP-only, and Joint MLM+NSP.

Supports multiple corpora: WikiText-2, OpenWebText, FineWeb-Edu, BookCorpus+Wikipedia.

Dataset registrations:
  MLM-only:    bert_mlm_wikitext2, bert_mlm_openwebtext, bert_mlm_fineweb
  NSP-only:    bert_nsp_wikitext2, bert_nsp_openwebtext, bert_nsp_fineweb
  Joint:       bert_pretrain_wikitext2, bert_pretrain_openwebtext,
               bert_pretrain_fineweb, bert_pretrain_bookcorpus_wiki
"""

import os
import random
import numpy as np
from experiment.registry import register_dataset, _maybe_distributed_sampler


# =============================================================================
# Dataset wrappers (module-level for pickling)
# =============================================================================

class _BertMLMDataset:
    """MLM-only dataset. Returns (input_ids, attention_mask, mlm_labels, token_type_ids)."""
    def __init__(self, token_chunks, tokenizer, mask_prob=0.15):
        self.chunks = token_chunks
        self.vocab_size = tokenizer.vocab_size
        self.mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
        self.cls_id = tokenizer.cls_token_id or 101
        self.sep_id = tokenizer.sep_token_id or 102
        self.pad_id = tokenizer.pad_token_id or 0
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        import torch
        tokens = self.chunks[idx].clone() if hasattr(self.chunks[idx], 'clone') else torch.tensor(self.chunks[idx], dtype=torch.long)
        seq_len = len(tokens)

        # Dynamic masking
        labels = torch.full((seq_len,), -100, dtype=torch.long)
        mask_candidates = torch.ones(seq_len, dtype=torch.bool)
        # Don't mask CLS/SEP/PAD
        mask_candidates[0] = False  # CLS
        for i in range(seq_len):
            if tokens[i].item() in (self.sep_id, self.pad_id):
                mask_candidates[i] = False

        n_mask = max(1, int(mask_candidates.sum().item() * self.mask_prob))
        mask_indices = mask_candidates.nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(mask_indices))[:n_mask]
        selected = mask_indices[perm]

        labels[selected] = tokens[selected]

        # 80% [MASK], 10% random, 10% keep
        rand = torch.rand(n_mask)
        mask_token_mask = rand < 0.8
        random_token_mask = (rand >= 0.8) & (rand < 0.9)

        tokens[selected[mask_token_mask]] = self.mask_id
        random_tokens = torch.randint(0, self.vocab_size, (random_token_mask.sum(),))
        tokens[selected[random_token_mask]] = random_tokens

        attention_mask = (tokens != self.pad_id).long()
        token_type_ids = torch.zeros(seq_len, dtype=torch.long)

        return tokens, attention_mask, labels, token_type_ids


class _BertNSPDataset:
    """NSP-only dataset. Returns (input_ids, attention_mask, token_type_ids, nsp_labels)."""
    def __init__(self, sentences, tokenizer, max_seq_len):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cls_id = tokenizer.cls_token_id or 101
        self.sep_id = tokenizer.sep_token_id or 102
        self.pad_id = tokenizer.pad_token_id or 0

    def __len__(self):
        return len(self.sentences) - 1

    def __getitem__(self, idx):
        import torch
        sent_a = self.sentences[idx]

        # 50% real next, 50% random
        if random.random() < 0.5 and idx + 1 < len(self.sentences):
            sent_b = self.sentences[idx + 1]
            nsp_label = 0  # IsNext
        else:
            rand_idx = random.randint(0, len(self.sentences) - 1)
            sent_b = self.sentences[rand_idx]
            nsp_label = 1  # NotNext

        # Truncate to fit: [CLS] sent_A [SEP] sent_B [SEP]
        max_tokens = self.max_seq_len - 3
        half = max_tokens // 2
        if len(sent_a) > half:
            sent_a = sent_a[:half]
        if len(sent_b) > max_tokens - len(sent_a):
            sent_b = sent_b[:max_tokens - len(sent_a)]

        input_ids = [self.cls_id] + sent_a + [self.sep_id] + sent_b + [self.sep_id]
        token_type_ids = [0] * (len(sent_a) + 2) + [1] * (len(sent_b) + 1)

        # Pad
        pad_len = self.max_seq_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.pad_id] * pad_len
        token_type_ids = token_type_ids + [0] * pad_len

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
                torch.tensor(token_type_ids, dtype=torch.long),
                torch.tensor(nsp_label, dtype=torch.long))


class _BertPreTrainDataset:
    """Joint MLM+NSP dataset. Returns dict with input_ids, attention_mask, token_type_ids, mlm_labels, nsp_labels."""
    def __init__(self, sentences, tokenizer, max_seq_len, mask_prob=0.15):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.vocab_size = tokenizer.vocab_size
        self.mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
        self.cls_id = tokenizer.cls_token_id or 101
        self.sep_id = tokenizer.sep_token_id or 102
        self.pad_id = tokenizer.pad_token_id or 0

    def __len__(self):
        return len(self.sentences) - 1

    def __getitem__(self, idx):
        import torch
        sent_a = list(self.sentences[idx])

        # 50% real next, 50% random
        if random.random() < 0.5 and idx + 1 < len(self.sentences):
            sent_b = list(self.sentences[idx + 1])
            nsp_label = 0
        else:
            rand_idx = random.randint(0, len(self.sentences) - 1)
            sent_b = list(self.sentences[rand_idx])
            nsp_label = 1

        # Truncate
        max_tokens = self.max_seq_len - 3
        half = max_tokens // 2
        if len(sent_a) > half:
            sent_a = sent_a[:half]
        if len(sent_b) > max_tokens - len(sent_a):
            sent_b = sent_b[:max_tokens - len(sent_a)]

        input_ids = [self.cls_id] + sent_a + [self.sep_id] + sent_b + [self.sep_id]
        token_type_ids = [0] * (len(sent_a) + 2) + [1] * (len(sent_b) + 1)
        seq_len = len(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.long)

        # Dynamic masking
        mlm_labels = torch.full((seq_len,), -100, dtype=torch.long)
        mask_candidates = torch.ones(seq_len, dtype=torch.bool)
        mask_candidates[0] = False  # CLS
        for i in range(seq_len):
            if input_ids[i].item() in (self.sep_id, self.pad_id):
                mask_candidates[i] = False

        n_mask = max(1, int(mask_candidates.sum().item() * self.mask_prob))
        mask_indices = mask_candidates.nonzero(as_tuple=True)[0]
        if len(mask_indices) > 0:
            perm = torch.randperm(len(mask_indices))[:n_mask]
            selected = mask_indices[perm]
            mlm_labels[selected] = input_ids[selected]

            rand = torch.rand(len(selected))
            mask_token_mask = rand < 0.8
            random_token_mask = (rand >= 0.8) & (rand < 0.9)

            input_ids[selected[mask_token_mask]] = self.mask_id
            random_tokens = torch.randint(0, self.vocab_size, (random_token_mask.sum(),))
            input_ids[selected[random_token_mask]] = random_tokens

        # Pad
        pad_len = self.max_seq_len - seq_len
        attention_mask = torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_id, dtype=torch.long)])
        token_type_ids_t = torch.cat([token_type_ids_t, torch.zeros(pad_len, dtype=torch.long)])
        mlm_labels = torch.cat([mlm_labels, torch.full((pad_len,), -100, dtype=torch.long)])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids_t,
            'mlm_labels': mlm_labels,
            'nsp_labels': torch.tensor(nsp_label, dtype=torch.long),
        }


class _NpBertMLMDataset:
    """Numpy-backend MLM dataset."""
    def __init__(self, token_chunks, tokenizer, mask_prob=0.15):
        self.chunks = token_chunks
        self.vocab_size = tokenizer.vocab_size
        self.mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
        self.pad_id = tokenizer.pad_token_id or 0
        self.sep_id = tokenizer.sep_token_id or 102
        self.mask_prob = mask_prob
        self.samples = list(range(len(token_chunks)))

    def __len__(self):
        return len(self.chunks)

    def load_sample(self, idx):
        tokens = np.array(self.chunks[idx], dtype=np.int64).copy()
        seq_len = len(tokens)

        labels = np.full(seq_len, -100, dtype=np.int64)
        mask_candidates = np.ones(seq_len, dtype=bool)
        mask_candidates[0] = False
        for i in range(seq_len):
            if tokens[i] in (self.sep_id, self.pad_id):
                mask_candidates[i] = False

        n_mask = max(1, int(mask_candidates.sum() * self.mask_prob))
        indices = np.where(mask_candidates)[0]
        selected = np.random.choice(indices, min(n_mask, len(indices)), replace=False)

        labels[selected] = tokens[selected]

        rand = np.random.rand(len(selected))
        tokens[selected[rand < 0.8]] = self.mask_id
        random_mask = (rand >= 0.8) & (rand < 0.9)
        tokens[selected[random_mask]] = np.random.randint(0, self.vocab_size, random_mask.sum())

        attention_mask = (tokens != self.pad_id).astype(np.int64)
        token_type_ids = np.zeros(seq_len, dtype=np.int64)

        return (tokens, attention_mask, labels, token_type_ids)


def _pretrain_collate_fn(batch):
    """Collate for BertPreTrainDataset (dict batches)."""
    import torch
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        if batch[0][k].dim() == 0:
            collated[k] = torch.stack([b[k] for b in batch])
        else:
            collated[k] = torch.stack([b[k] for b in batch])
    return collated


# =============================================================================
# Helpers
# =============================================================================

def _split_into_sentences(texts, tokenizer, min_len=5):
    """Tokenize texts and split into sentence-level token lists."""
    sentences = []
    for text in texts:
        if not text.strip():
            continue
        # Simple sentence splitting by periods, then tokenize
        parts = text.replace('\n', ' ').split('.')
        for part in parts:
            part = part.strip()
            if len(part) < 10:
                continue
            tokens = tokenizer.encode(part, add_special_tokens=False)
            if len(tokens) >= min_len:
                sentences.append(tokens)
    return sentences


def _make_mlm_chunks(texts, tokenizer, max_seq_len, subset=None):
    """Tokenize texts into [CLS] tokens [SEP] chunks for MLM."""
    import torch

    cls_id = tokenizer.cls_token_id or 101
    sep_id = tokenizer.sep_token_id or 102

    all_tokens = []
    for text in texts:
        if text.strip():
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

    # Chunk into max_seq_len - 2 tokens, adding CLS/SEP
    chunk_size = max_seq_len - 2
    n_chunks = len(all_tokens) // chunk_size
    if n_chunks == 0:
        n_chunks = 1
        all_tokens = all_tokens + [tokenizer.pad_token_id or 0] * (chunk_size - len(all_tokens))

    chunks = []
    for i in range(n_chunks):
        chunk = [cls_id] + all_tokens[i * chunk_size:(i + 1) * chunk_size] + [sep_id]
        chunks.append(torch.tensor(chunk, dtype=torch.long))

    if subset and subset < len(chunks):
        chunks = chunks[:subset]

    return chunks


def _load_texts_from_dataset(dataset_name, config, max_docs=None):
    """Load raw text from a HuggingFace dataset."""
    from datasets import load_dataset

    if dataset_name == 'wikitext2':
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
        texts = ds['train']['text']
    elif dataset_name == 'openwebtext':
        ds = load_dataset('openwebtext', split='train', streaming=True)
        limit = max_docs or config.subset or 100_000
        texts = []
        for i, ex in enumerate(ds):
            if i >= limit:
                break
            texts.append(ex['text'])
    elif dataset_name == 'fineweb':
        ds = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT',
                          split='train', streaming=True)
        limit = max_docs or config.subset or 100_000
        texts = []
        for i, ex in enumerate(ds):
            if i >= limit:
                break
            texts.append(ex['text'])
    elif dataset_name == 'bookcorpus_wiki':
        ds_book = load_dataset('bookcorpus', split='train', streaming=True)
        ds_wiki = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)
        limit = (max_docs or config.subset or 100_000) // 2
        texts = []
        for i, ex in enumerate(ds_book):
            if i >= limit:
                break
            texts.append(ex['text'])
        for i, ex in enumerate(ds_wiki):
            if i >= limit:
                break
            texts.append(ex['text'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return texts


def _build_bert_loaders(dataset, config, collate_fn=None):
    """Build train/val/test loaders from a single dataset."""
    import torch
    from torch.utils.data import DataLoader

    n = len(dataset)
    n_val = max(1, int(n * 0.05))
    n_test = max(1, int(n * 0.05))
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.seed),
    )

    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
              persistent_workers=config.num_workers > 0,
              prefetch_factor=4 if config.num_workers > 0 else None)
    if collate_fn:
        kw['collate_fn'] = collate_fn

    train_sampler = _maybe_distributed_sampler(train_ds, config, shuffle=True)
    print(f"  BERT dataset: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    return (
        DataLoader(train_ds, config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **kw),
        DataLoader(val_ds, config.batch_size, shuffle=False, **kw),
        DataLoader(test_ds, config.batch_size, shuffle=False, **kw),
    )


# =============================================================================
# MLM-only datasets
# =============================================================================

def _build_mlm_dataset(corpus_name, config):
    from transformers import AutoTokenizer
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    texts = _load_texts_from_dataset(corpus_name, config)
    chunks = _make_mlm_chunks(texts, tokenizer, config.max_seq_len, config.subset)
    dataset = _BertMLMDataset(chunks, tokenizer)
    return _build_bert_loaders(dataset, config)


@register_dataset('bert_mlm_wikitext2', 'pytorch')
def _pt_bert_mlm_wikitext2(config):
    return _build_mlm_dataset('wikitext2', config)

@register_dataset('bert_mlm_openwebtext', 'pytorch')
def _pt_bert_mlm_openwebtext(config):
    return _build_mlm_dataset('openwebtext', config)

@register_dataset('bert_mlm_fineweb', 'pytorch')
def _pt_bert_mlm_fineweb(config):
    return _build_mlm_dataset('fineweb', config)


# =============================================================================
# NSP-only datasets
# =============================================================================

def _build_nsp_dataset(corpus_name, config):
    from transformers import AutoTokenizer
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    texts = _load_texts_from_dataset(corpus_name, config)
    sentences = _split_into_sentences(texts, tokenizer)
    if config.subset and config.subset < len(sentences):
        sentences = sentences[:config.subset]
    dataset = _BertNSPDataset(sentences, tokenizer, config.max_seq_len)
    return _build_bert_loaders(dataset, config)


@register_dataset('bert_nsp_wikitext2', 'pytorch')
def _pt_bert_nsp_wikitext2(config):
    return _build_nsp_dataset('wikitext2', config)

@register_dataset('bert_nsp_openwebtext', 'pytorch')
def _pt_bert_nsp_openwebtext(config):
    return _build_nsp_dataset('openwebtext', config)

@register_dataset('bert_nsp_fineweb', 'pytorch')
def _pt_bert_nsp_fineweb(config):
    return _build_nsp_dataset('fineweb', config)


# =============================================================================
# Joint MLM+NSP datasets
# =============================================================================

def _build_pretrain_dataset(corpus_name, config):
    from transformers import AutoTokenizer
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    texts = _load_texts_from_dataset(corpus_name, config)
    sentences = _split_into_sentences(texts, tokenizer)
    if config.subset and config.subset < len(sentences):
        sentences = sentences[:config.subset]
    dataset = _BertPreTrainDataset(sentences, tokenizer, config.max_seq_len)
    return _build_bert_loaders(dataset, config, collate_fn=_pretrain_collate_fn)


@register_dataset('bert_pretrain_wikitext2', 'pytorch')
def _pt_bert_pretrain_wikitext2(config):
    return _build_pretrain_dataset('wikitext2', config)

@register_dataset('bert_pretrain_openwebtext', 'pytorch')
def _pt_bert_pretrain_openwebtext(config):
    return _build_pretrain_dataset('openwebtext', config)

@register_dataset('bert_pretrain_fineweb', 'pytorch')
def _pt_bert_pretrain_fineweb(config):
    return _build_pretrain_dataset('fineweb', config)

@register_dataset('bert_pretrain_bookcorpus_wiki', 'pytorch')
def _pt_bert_pretrain_bookcorpus_wiki(config):
    return _build_pretrain_dataset('bookcorpus_wiki', config)


# =============================================================================
# Numpy MLM dataset (for mac testing)
# =============================================================================

@register_dataset('bert_mlm_wikitext2', 'numpy')
def _np_bert_mlm_wikitext2(config):
    from python.utils.data_utils import DataLoader as NpLoader
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    texts = _load_texts_from_dataset('wikitext2', config)
    chunks = _make_mlm_chunks(texts, tokenizer, config.max_seq_len, config.subset)

    # Convert chunks to numpy
    np_chunks = [c.numpy() for c in chunks]
    dataset = _NpBertMLMDataset(np_chunks, tokenizer)

    n = len(dataset)
    n_val = max(1, int(n * 0.05))
    n_test = max(1, int(n * 0.05))

    np.random.seed(config.seed)
    idx = np.random.permutation(n)
    test_indices = idx[:n_test]
    val_indices = idx[n_test:n_test + n_val]
    train_indices = idx[n_test + n_val:]

    class _SubsetWrapper:
        def __init__(self, parent, indices):
            self.parent = parent
            self.indices = indices
            self.samples = list(range(len(indices)))
        def __len__(self):
            return len(self.indices)
        def load_sample(self, idx):
            return self.parent.load_sample(self.indices[idx])

    print(f"  BERT MLM (numpy): {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    return (
        NpLoader(_SubsetWrapper(dataset, train_indices), batch_size=config.batch_size, shuffle=True),
        NpLoader(_SubsetWrapper(dataset, val_indices), batch_size=config.batch_size, shuffle=False),
        NpLoader(_SubsetWrapper(dataset, test_indices), batch_size=config.batch_size, shuffle=False),
    )


@register_dataset('bert_pretrain_wikitext2', 'numpy')
def _np_bert_pretrain_wikitext2(config):
    """Numpy backend: use MLM-only dataset as fallback for pretrain testing."""
    return _np_bert_mlm_wikitext2(config)
