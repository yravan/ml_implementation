"""POS Tagging Dataset — Universal Dependencies English Web Treebank."""

import os
import numpy as np
from experiment.registry import register_dataset, _maybe_distributed_sampler


class _POSDataset:
    """PyTorch POS tagging dataset. Returns (input_ids, attention_mask, labels)."""
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]


class _NpPOSDataset:
    """Numpy POS tagging dataset."""
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.samples = list(range(len(input_ids)))

    def __len__(self):
        return len(self.input_ids)

    def load_sample(self, idx):
        return (self.input_ids[idx], self.attention_masks[idx], self.labels[idx])


def _process_ud_pos(config, return_numpy=False):
    """Download and process Universal Dependencies POS tagging data."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    max_seq_len = config.max_seq_len

    ds = load_dataset('universal_dependencies', 'en_ewt', trust_remote_code=True)

    # Universal POS tags — dataset uses ClassLabel with 18 tags (0..17)
    # The integer labels from the dataset are already correct indices
    num_tags = len(ds['train'].features['upos'].feature.names)  # typically 18

    def process_split(split):
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for example in split:
            words = example['tokens']
            pos_tags = example['upos']

            # Tokenize word by word, align labels
            input_ids = [tokenizer.cls_token_id]
            labels = [-100]

            for word, tag in zip(words, pos_tags):
                word_tokens = tokenizer.encode(word, add_special_tokens=False)
                if len(word_tokens) == 0:
                    continue
                input_ids.extend(word_tokens)
                # First subword gets the tag, rest get -100
                labels.append(tag)
                labels.extend([-100] * (len(word_tokens) - 1))

            input_ids.append(tokenizer.sep_token_id)
            labels.append(-100)

            # Truncate
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                labels = labels[:max_seq_len]

            # Pad
            pad_len = max_seq_len - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

            if return_numpy:
                all_input_ids.append(np.array(input_ids, dtype=np.int64))
                all_attention_masks.append(np.array(attention_mask, dtype=np.int64))
                all_labels.append(np.array(labels, dtype=np.int64))
            else:
                import torch
                all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                all_attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))
                all_labels.append(torch.tensor(labels, dtype=torch.long))

        if return_numpy:
            return (np.stack(all_input_ids), np.stack(all_attention_masks), np.stack(all_labels))
        else:
            import torch
            return (torch.stack(all_input_ids), torch.stack(all_attention_masks), torch.stack(all_labels))

    return ds, process_split, num_tags


@register_dataset('ud_pos', 'pytorch')
def _pt_ud_pos(config):
    import torch
    from torch.utils.data import DataLoader

    ds, process_split, num_tags = _process_ud_pos(config, return_numpy=False)

    train_ids, train_masks, train_labels = process_split(ds['train'])
    val_ids, val_masks, val_labels = process_split(ds['validation'])
    test_ids, test_masks, test_labels = process_split(ds['test'])

    if config.subset and config.subset < len(train_ids):
        train_ids = train_ids[:config.subset]
        train_masks = train_masks[:config.subset]
        train_labels = train_labels[:config.subset]

    print(f"  UD POS: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test ({num_tags} tags)")

    kw = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
              persistent_workers=config.num_workers > 0,
              prefetch_factor=4 if config.num_workers > 0 else None)

    train_ds = _POSDataset(train_ids, train_masks, train_labels)
    train_sampler = _maybe_distributed_sampler(train_ds, config, shuffle=True)
    return (
        DataLoader(train_ds, config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, **kw),
        DataLoader(_POSDataset(val_ids, val_masks, val_labels), config.batch_size, shuffle=False, **kw),
        DataLoader(_POSDataset(test_ids, test_masks, test_labels), config.batch_size, shuffle=False, **kw),
    )


@register_dataset('ud_pos', 'numpy')
def _np_ud_pos(config):
    from python.utils.data_utils import DataLoader as NpLoader

    ds, process_split, num_tags = _process_ud_pos(config, return_numpy=True)

    train_ids, train_masks, train_labels = process_split(ds['train'])
    val_ids, val_masks, val_labels = process_split(ds['validation'])
    test_ids, test_masks, test_labels = process_split(ds['test'])

    if config.subset and config.subset < len(train_ids):
        train_ids = train_ids[:config.subset]
        train_masks = train_masks[:config.subset]
        train_labels = train_labels[:config.subset]

    print(f"  UD POS (numpy): {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test ({num_tags} tags)")

    return (
        NpLoader(_NpPOSDataset(train_ids, train_masks, train_labels), batch_size=config.batch_size, shuffle=True),
        NpLoader(_NpPOSDataset(val_ids, val_masks, val_labels), batch_size=config.batch_size, shuffle=False),
        NpLoader(_NpPOSDataset(test_ids, test_masks, test_labels), batch_size=config.batch_size, shuffle=False),
    )
