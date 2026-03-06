"""Language modeling datasets: WikiText-2, WikiText-103, OpenWebText."""

import os
from experiment.registry import register_dataset, _maybe_distributed_sampler


class _TokenizedTextDataset:
    """Wraps tokenized text chunks as a PyTorch dataset returning (input_ids,)."""
    def __init__(self, token_ids):
        import torch
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.token_ids = token_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return (self.token_ids[idx],)


class _ArrowWrapper:
    """Thin wrapper around a HuggingFace Dataset to return (input_ids,) tuples."""
    def __init__(self, hf_dataset):
        self.ds = hf_dataset
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        return (self.ds[idx]['input_ids'],)


class _NpTokenizedTextDataset:
    """Numpy-backend dataset for tokenized text chunks. Returns (input_ids,)."""
    def __init__(self, token_ids):
        import numpy as np
        if not isinstance(token_ids, np.ndarray):
            token_ids = np.array(token_ids, dtype=np.int64)
        self.token_ids = token_ids
        self.samples = list(range(len(token_ids)))

    def __len__(self):
        return len(self.token_ids)

    def load_sample(self, idx):
        return (self.token_ids[idx],)


def _tokenize_and_chunk(texts, tokenizer, max_seq_len, subset=None):
    """Tokenize a list of texts and chunk into fixed-length sequences."""
    import torch

    all_tokens = []
    for text in texts:
        if text.strip():
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

    n_chunks = len(all_tokens) // max_seq_len
    if n_chunks == 0:
        all_tokens = all_tokens + [tokenizer.eos_token_id or 0] * (max_seq_len - len(all_tokens))
        n_chunks = 1

    all_tokens = all_tokens[:n_chunks * max_seq_len]
    chunks = torch.tensor(all_tokens, dtype=torch.long).reshape(n_chunks, max_seq_len)

    if subset and subset < len(chunks):
        chunks = chunks[:subset]

    return chunks


# =============================================================================
# PyTorch
# =============================================================================

@register_dataset('wikitext2', 'pytorch')
def _pt_wikitext2(config):
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset('wikitext', 'wikitext-2-raw-v1')

    train_chunks = _tokenize_and_chunk(ds['train']['text'], tokenizer, config.max_seq_len, config.subset)
    val_chunks = _tokenize_and_chunk(ds['validation']['text'], tokenizer, config.max_seq_len)
    test_chunks = _tokenize_and_chunk(ds['test']['text'], tokenizer, config.max_seq_len)

    print(f"  WikiText-2: {len(train_chunks)} train, {len(val_chunks)} val, {len(test_chunks)} test sequences")

    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
              persistent_workers=config.num_workers > 0,
              prefetch_factor=4 if config.num_workers > 0 else None)

    train_ds = _TokenizedTextDataset(train_chunks)
    train_sampler = _maybe_distributed_sampler(train_ds, config, shuffle=True)
    return (
        DataLoader(train_ds, config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **kw),
        DataLoader(_TokenizedTextDataset(val_chunks), config.batch_size, shuffle=False, **kw),
        DataLoader(_TokenizedTextDataset(test_chunks), config.batch_size, shuffle=False, **kw),
    )


@register_dataset('wikitext103', 'pytorch')
def _pt_wikitext103(config):
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset('wikitext', 'wikitext-103-raw-v1')

    train_chunks = _tokenize_and_chunk(ds['train']['text'], tokenizer, config.max_seq_len, config.subset)
    val_chunks = _tokenize_and_chunk(ds['validation']['text'], tokenizer, config.max_seq_len)
    test_chunks = _tokenize_and_chunk(ds['test']['text'], tokenizer, config.max_seq_len)

    print(f"  WikiText-103: {len(train_chunks)} train, {len(val_chunks)} val, {len(test_chunks)} test sequences")

    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
              persistent_workers=config.num_workers > 0,
              prefetch_factor=4 if config.num_workers > 0 else None)

    train_ds = _TokenizedTextDataset(train_chunks)
    train_sampler = _maybe_distributed_sampler(train_ds, config, shuffle=True)
    return (
        DataLoader(train_ds, config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **kw),
        DataLoader(_TokenizedTextDataset(val_chunks), config.batch_size, shuffle=False, **kw),
        DataLoader(_TokenizedTextDataset(test_chunks), config.batch_size, shuffle=False, **kw),
    )


@register_dataset('openwebtext', 'pytorch')
def _pt_openwebtext(config):
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import os

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, model_max_length=100000)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_len = config.max_seq_len
    hf_cache = '/tmp/hf_datasets'
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    eos_id = tokenizer.eos_token_id

    def tokenize_fn(examples):
        out = tokenizer(examples['text'], add_special_tokens=False)
        for ids in out['input_ids']:
            ids.append(eos_id)
        return out

    def group_texts(examples):
        from itertools import chain
        concatenated = list(chain.from_iterable(examples['input_ids']))
        total_length = (len(concatenated) // max_seq_len) * max_seq_len
        result = {'input_ids': [concatenated[i:i + max_seq_len]
                  for i in range(0, total_length, max_seq_len)]}
        return result

    def _load_and_process():
        ds = load_dataset('openwebtext', split='train', cache_dir=hf_cache)
        tokenized = ds.map(tokenize_fn, batched=True, batch_size=5000,
                           num_proc=os.cpu_count(), remove_columns=['text'],
                           desc="Tokenizing")
        chunked = tokenized.map(group_texts, batched=True, batch_size=5000,
                                num_proc=os.cpu_count(),
                                remove_columns=['input_ids', 'attention_mask'],
                                desc="Chunking")
        del ds, tokenized
        return chunked

    if rank == 0:
        print("  Downloading and tokenizing OpenWebText...")
        chunked = _load_and_process()

    if world_size > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    if rank != 0:
        chunked = _load_and_process()

    chunked.set_format('torch')

    split = chunked.train_test_split(test_size=0.05, seed=config.seed)
    val_test = split['test'].train_test_split(test_size=0.5, seed=config.seed)

    train_ds = _ArrowWrapper(split['train'])
    val_ds = _ArrowWrapper(val_test['train'])
    test_ds = _ArrowWrapper(val_test['test'])

    print(f"  OpenWebText: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test sequences")

    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
              persistent_workers=config.num_workers > 0,
              prefetch_factor=4 if config.num_workers > 0 else None)

    train_sampler = _maybe_distributed_sampler(train_ds, config, shuffle=True)
    return (
        DataLoader(train_ds, config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **kw),
        DataLoader(val_ds, config.batch_size, shuffle=False, **kw),
        DataLoader(test_ds, config.batch_size, shuffle=False, **kw),
    )


# =============================================================================
# Numpy
# =============================================================================

@register_dataset('wikitext2', 'numpy')
def _np_wikitext2(config):
    from python.utils.data_utils import DataLoader as NpLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
    max_seq_len = config.max_seq_len

    train_chunks = _tokenize_and_chunk(ds['train']['text'], tokenizer, max_seq_len, config.subset)
    val_chunks = _tokenize_and_chunk(ds['validation']['text'], tokenizer, max_seq_len)
    test_chunks = _tokenize_and_chunk(ds['test']['text'], tokenizer, max_seq_len)

    print(f"  WikiText-2 (numpy): {len(train_chunks)} train, {len(val_chunks)} val, {len(test_chunks)} test chunks")

    return (
        NpLoader(_NpTokenizedTextDataset(train_chunks), batch_size=config.batch_size, shuffle=True),
        NpLoader(_NpTokenizedTextDataset(val_chunks), batch_size=config.batch_size, shuffle=False),
        NpLoader(_NpTokenizedTextDataset(test_chunks), batch_size=config.batch_size, shuffle=False),
    )


@register_dataset('wikitext103', 'numpy')
def _np_wikitext103(config):
    from python.utils.data_utils import DataLoader as NpLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1')
    max_seq_len = config.max_seq_len

    train_chunks = _tokenize_and_chunk(ds['train']['text'], tokenizer, max_seq_len, config.subset)
    val_chunks = _tokenize_and_chunk(ds['validation']['text'], tokenizer, max_seq_len)
    test_chunks = _tokenize_and_chunk(ds['test']['text'], tokenizer, max_seq_len)

    print(f"  WikiText-103 (numpy): {len(train_chunks)} train, {len(val_chunks)} val, {len(test_chunks)} test chunks")

    return (
        NpLoader(_NpTokenizedTextDataset(train_chunks), batch_size=config.batch_size, shuffle=True),
        NpLoader(_NpTokenizedTextDataset(val_chunks), batch_size=config.batch_size, shuffle=False),
        NpLoader(_NpTokenizedTextDataset(test_chunks), batch_size=config.batch_size, shuffle=False),
    )


@register_dataset('openwebtext', 'numpy')
def _np_openwebtext(config):
    from python.utils.data_utils import DataLoader as NpLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    max_seq_len = config.max_seq_len
    ds = load_dataset('openwebtext', split='train', streaming=True)

    limit = config.subset or 100_000
    texts = []
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        texts.append(ex['text'])

    all_chunks = _tokenize_and_chunk(texts, tokenizer, max_seq_len)
    n = len(all_chunks)
    n_val = max(1, int(n * 0.05))
    n_test = max(1, int(n * 0.05))

    np.random.seed(config.seed)
    idx = np.random.permutation(n)
    test_chunks = [all_chunks[i] for i in idx[:n_test]]
    val_chunks = [all_chunks[i] for i in idx[n_test:n_test + n_val]]
    train_chunks = [all_chunks[i] for i in idx[n_test + n_val:]]

    print(f"  OpenWebText (numpy): {len(train_chunks)} train, {len(val_chunks)} val, {len(test_chunks)} test chunks")

    return (
        NpLoader(_NpTokenizedTextDataset(train_chunks), batch_size=config.batch_size, shuffle=True),
        NpLoader(_NpTokenizedTextDataset(val_chunks), batch_size=config.batch_size, shuffle=False),
        NpLoader(_NpTokenizedTextDataset(test_chunks), batch_size=config.batch_size, shuffle=False),
    )
