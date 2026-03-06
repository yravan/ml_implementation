"""Translation datasets: Multi30K, WMT14."""

import os
import functools
import numpy as np
from experiment.registry import register_dataset, _maybe_distributed_sampler


class _Seq2SeqArrowWrapper:
    """Wraps Arrow dataset with src_input_ids/tgt_input_ids columns."""
    def __init__(self, hf_dataset):
        self.ds = hf_dataset
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        row = self.ds[idx]
        return (row['src_input_ids'], row['tgt_input_ids'])


class _NpSeq2SeqDataset:
    """Numpy-backend dataset for parallel src/tgt token IDs."""
    def __init__(self, src_ids, tgt_ids):
        if not isinstance(src_ids, np.ndarray):
            src_ids = np.array(src_ids, dtype=np.int64)
        if not isinstance(tgt_ids, np.ndarray):
            tgt_ids = np.array(tgt_ids, dtype=np.int64)
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids
        self.samples = list(range(len(src_ids)))

    def __len__(self):
        return len(self.src_ids)

    def load_sample(self, idx):
        return (self.src_ids[idx], self.tgt_ids[idx])


def _seq2seq_collate_fn(batch, pad_id=0):
    """Pad each batch to longest-in-batch (dynamic padding)."""
    import torch
    src_list, tgt_list = zip(*batch)
    src_max = max(len(s) for s in src_list)
    tgt_max = max(len(t) for t in tgt_list)
    src_padded = torch.full((len(batch), src_max), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max), pad_id, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        s_t = s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.long)
        t_t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long)
        src_padded[i, :len(s_t)] = s_t
        tgt_padded[i, :len(t_t)] = t_t
    return src_padded, tgt_padded


def _tokenize_parallel(src_texts, tgt_texts, tokenizer, max_seq_len):
    """Batch-tokenize parallel src/tgt texts into fixed-length numpy arrays."""
    src_enc = tokenizer(src_texts, truncation=True, padding='max_length',
                        max_length=max_seq_len, return_tensors='np')
    tgt_enc = tokenizer(tgt_texts, truncation=True, padding='max_length',
                        max_length=max_seq_len, return_tensors='np')
    return src_enc['input_ids'].astype(np.int64), tgt_enc['input_ids'].astype(np.int64)


# =============================================================================
# PyTorch
# =============================================================================

@register_dataset('multi30k', 'pytorch')
def _pt_multi30k(config):
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    max_seq_len = config.max_seq_len
    ds = load_dataset('bentrevett/multi30k')

    def tokenize_fn(examples):
        src = tokenizer(examples['en'], truncation=True, max_length=max_seq_len)
        tgt = tokenizer(examples['de'], truncation=True, max_length=max_seq_len)
        return {'src_input_ids': src['input_ids'], 'tgt_input_ids': tgt['input_ids']}

    for split_name in ds:
        ds[split_name] = ds[split_name].map(
            tokenize_fn, batched=True, num_proc=os.cpu_count(),
            remove_columns=ds[split_name].column_names,
            desc=f"Tokenizing {split_name}",
        )

    if config.subset and config.subset < len(ds['train']):
        ds['train'] = ds['train'].select(range(config.subset))

    train_ds = _Seq2SeqArrowWrapper(ds['train'])
    val_ds = _Seq2SeqArrowWrapper(ds['validation'])
    test_ds = _Seq2SeqArrowWrapper(ds['test'])

    print(f"  Multi30k: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test pairs")

    collate = functools.partial(_seq2seq_collate_fn, pad_id=pad_token_id)
    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
              persistent_workers=config.num_workers > 0,
              prefetch_factor=4 if config.num_workers > 0 else None,
              collate_fn=collate)

    train_sampler = _maybe_distributed_sampler(train_ds, config, shuffle=True)
    return (
        DataLoader(train_ds, config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **kw),
        DataLoader(val_ds, config.batch_size, shuffle=False, **kw),
        DataLoader(test_ds, config.batch_size, shuffle=False, **kw),
    )


@register_dataset('wmt14', 'pytorch')
def _pt_wmt14(config):
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    max_seq_len = config.max_seq_len
    hf_cache = '/tmp/hf_datasets'
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    def tokenize_fn(examples):
        src_texts = [t['en'] for t in examples['translation']]
        tgt_texts = [t['de'] for t in examples['translation']]
        src = tokenizer(src_texts, truncation=True, max_length=max_seq_len)
        tgt = tokenizer(tgt_texts, truncation=True, max_length=max_seq_len)
        return {'src_input_ids': src['input_ids'], 'tgt_input_ids': tgt['input_ids']}

    def _load_and_process():
        ds = load_dataset('wmt14', 'de-en', cache_dir=hf_cache)
        for split_name in ds:
            ds[split_name] = ds[split_name].map(
                tokenize_fn, batched=True, batch_size=5000,
                num_proc=os.cpu_count(),
                remove_columns=ds[split_name].column_names,
                desc=f"Tokenizing {split_name}",
            )
        return ds

    if rank == 0:
        print("  Downloading and tokenizing WMT14...")
        ds = _load_and_process()

    if world_size > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    if rank != 0:
        ds = _load_and_process()

    if config.subset and config.subset < len(ds['train']):
        ds['train'] = ds['train'].select(range(config.subset))

    train_ds = _Seq2SeqArrowWrapper(ds['train'])
    val_ds = _Seq2SeqArrowWrapper(ds['validation'])
    test_ds = _Seq2SeqArrowWrapper(ds['test'])

    print(f"  WMT14: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test pairs")

    collate = functools.partial(_seq2seq_collate_fn, pad_id=pad_token_id)
    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
              persistent_workers=config.num_workers > 0,
              prefetch_factor=4 if config.num_workers > 0 else None,
              collate_fn=collate)

    train_sampler = _maybe_distributed_sampler(train_ds, config, shuffle=True)
    return (
        DataLoader(train_ds, config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **kw),
        DataLoader(val_ds, config.batch_size, shuffle=False, **kw),
        DataLoader(test_ds, config.batch_size, shuffle=False, **kw),
    )


# =============================================================================
# Numpy
# =============================================================================

@register_dataset('multi30k', 'numpy')
def _np_multi30k(config):
    from python.utils.data_utils import DataLoader as NpLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_len = config.max_seq_len
    ds = load_dataset('bentrevett/multi30k')

    def process_split(split, subset=None):
        src_texts = [ex['en'] for ex in split]
        tgt_texts = [ex['de'] for ex in split]
        if subset:
            src_texts = src_texts[:subset]
            tgt_texts = tgt_texts[:subset]
        return _tokenize_parallel(src_texts, tgt_texts, tokenizer, max_seq_len)

    train_src, train_tgt = process_split(ds['train'], config.subset)
    val_src, val_tgt = process_split(ds['validation'])
    test_src, test_tgt = process_split(ds['test'])

    print(f"  Multi30k (numpy): {len(train_src)} train, {len(val_src)} val, {len(test_src)} test pairs")

    return (
        NpLoader(_NpSeq2SeqDataset(train_src, train_tgt), batch_size=config.batch_size, shuffle=True),
        NpLoader(_NpSeq2SeqDataset(val_src, val_tgt), batch_size=config.batch_size, shuffle=False),
        NpLoader(_NpSeq2SeqDataset(test_src, test_tgt), batch_size=config.batch_size, shuffle=False),
    )


@register_dataset('wmt14', 'numpy')
def _np_wmt14(config):
    from python.utils.data_utils import DataLoader as NpLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_len = config.max_seq_len
    ds = load_dataset('wmt14', 'de-en')

    def process_split(split, subset=None):
        src_texts = [ex['translation']['en'] for ex in split]
        tgt_texts = [ex['translation']['de'] for ex in split]
        if subset:
            src_texts = src_texts[:subset]
            tgt_texts = tgt_texts[:subset]
        return _tokenize_parallel(src_texts, tgt_texts, tokenizer, max_seq_len)

    train_src, train_tgt = process_split(ds['train'], config.subset)
    val_src, val_tgt = process_split(ds['validation'])
    test_src, test_tgt = process_split(ds['test'])

    print(f"  WMT14 (numpy): {len(train_src)} train, {len(val_src)} val, {len(test_src)} test pairs")

    return (
        NpLoader(_NpSeq2SeqDataset(train_src, train_tgt), batch_size=config.batch_size, shuffle=True),
        NpLoader(_NpSeq2SeqDataset(val_src, val_tgt), batch_size=config.batch_size, shuffle=False),
        NpLoader(_NpSeq2SeqDataset(test_src, test_tgt), batch_size=config.batch_size, shuffle=False),
    )
