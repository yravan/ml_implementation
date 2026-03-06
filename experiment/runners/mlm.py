"""MLM Runner — Masked Language Modeling training loop."""

import math
from contextlib import nullcontext

from experiment.runner import BaseRunner


class MLMRunner(BaseRunner):
    """
    Masked Language Modeling runner.

    Batch format: (input_ids, attention_mask, mlm_labels, token_type_ids)
    Loss: CrossEntropyLoss(ignore_index=-100) on [B*T, V]
    Metrics: MLM accuracy (masked positions only), perplexity
    """

    def setup_metrics(self):
        def mlm_accuracy(logits, targets):
            """Accuracy on masked positions only (where targets != -100)."""
            if logits is None or targets is None:
                return 0.0
            if hasattr(logits, 'detach'):
                mask = (targets != -100)
                if mask.sum() == 0:
                    return 0.0
                preds = logits.argmax(-1)
                return (preds[mask] == targets[mask]).float().sum().item()
            return 0.0
        return {'mlm_accuracy': mlm_accuracy}

    def setup_criterion(self):
        if self.config.backend == 'pytorch':
            import torch.nn.functional as F
            def mlm_criterion(logits, targets):
                B, T, V = logits.shape
                return F.cross_entropy(
                    logits.reshape(B * T, V), targets.reshape(B * T),
                    ignore_index=-100,
                )
            return mlm_criterion
        from python.optimization.losses import CrossEntropyLoss
        ce = CrossEntropyLoss()
        def np_mlm_criterion(logits, targets, reduction='mean'):
            return ce(logits, targets, ignore_index=-100, reduction=reduction)
        return np_mlm_criterion

    def train_step(self, model, batch, criterion, optimizer, is_accumulating=False):
        if self.config.backend == 'numpy':
            return self._np_train_step(model, batch, criterion, optimizer)

        import torch
        backend = self.backend
        grad_accum = self.config.gradient_accumulation_steps

        input_ids = batch[0].to(backend.device, non_blocking=True)
        attention_mask = batch[1].to(backend.device, non_blocking=True)
        mlm_labels = batch[2].to(backend.device, non_blocking=True)
        token_type_ids = batch[3].to(backend.device, non_blocking=True)

        ctx = model.no_sync() if (is_accumulating and hasattr(model, 'no_sync')) else nullcontext()

        with ctx:
            if backend._use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                    loss = criterion(logits, mlm_labels)
                    if grad_accum > 1:
                        loss = loss / grad_accum
                backend._scaler.scale(loss).backward()
            else:
                logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                loss = criterion(logits, mlm_labels)
                if grad_accum > 1:
                    loss = loss / grad_accum
                loss.backward()

        raw_loss = loss.detach() * grad_accum if grad_accum > 1 else loss.detach()
        return logits.detach(), raw_loss, input_ids.shape[0], mlm_labels

    def _np_train_step(self, model, batch, criterion, optimizer):
        from python.foundations.computational_graph import Tensor
        import numpy as np

        input_ids = batch[0]
        if hasattr(input_ids, 'numpy'): input_ids = input_ids.numpy()
        attention_mask = batch[1]
        if hasattr(attention_mask, 'numpy'): attention_mask = attention_mask.numpy()
        mlm_labels = batch[2]
        if hasattr(mlm_labels, 'numpy'): mlm_labels = mlm_labels.numpy()
        token_type_ids = batch[3]
        if hasattr(token_type_ids, 'numpy'): token_type_ids = token_type_ids.numpy()

        x = Tensor(input_ids, requires_grad=True)
        tt = Tensor(token_type_ids)
        am = Tensor(attention_mask)

        logits = model(x, token_type_ids=tt, attention_mask=am)

        B, T = mlm_labels.shape
        logits_flat = logits.reshape(B * T, -1)
        targets_flat = Tensor(mlm_labels.reshape(B * T))

        loss = criterion(logits_flat, targets_flat, reduction='mean')
        loss.backward()

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return logits, batch_loss, B, mlm_labels

    def eval_step(self, model, batch, criterion):
        if self.config.backend == 'numpy':
            return self._np_eval_step(model, batch, criterion)

        import torch
        backend = self.backend
        input_ids = batch[0].to(backend.device, non_blocking=True)
        attention_mask = batch[1].to(backend.device, non_blocking=True)
        mlm_labels = batch[2].to(backend.device, non_blocking=True)
        token_type_ids = batch[3].to(backend.device, non_blocking=True)

        if backend._use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                loss = criterion(logits, mlm_labels)
        else:
            logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss = criterion(logits, mlm_labels)

        return logits.detach(), loss.detach(), input_ids.shape[0], mlm_labels

    def _np_eval_step(self, model, batch, criterion):
        from python.foundations.computational_graph import Tensor

        input_ids = batch[0]
        if hasattr(input_ids, 'numpy'): input_ids = input_ids.numpy()
        attention_mask = batch[1]
        if hasattr(attention_mask, 'numpy'): attention_mask = attention_mask.numpy()
        mlm_labels = batch[2]
        if hasattr(mlm_labels, 'numpy'): mlm_labels = mlm_labels.numpy()
        token_type_ids = batch[3]
        if hasattr(token_type_ids, 'numpy'): token_type_ids = token_type_ids.numpy()

        x = Tensor(input_ids, requires_grad=False)
        tt = Tensor(token_type_ids)
        am = Tensor(attention_mask)

        logits = model(x, token_type_ids=tt, attention_mask=am)

        B, T = mlm_labels.shape
        logits_flat = logits.reshape(B * T, -1)
        targets_flat = Tensor(mlm_labels.reshape(B * T))

        loss = criterion(logits_flat, targets_flat, reduction='mean')

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return logits, batch_loss, B, mlm_labels

    def format_metric(self, name, value):
        if name == 'loss':
            return f"{value:.4f}"
        if name == 'perplexity':
            return f"{value:.2f}"
        if name == 'mlm_accuracy':
            return f"{value*100:.2f}%"
        return f"{value:.4f}"

    def _evaluate(self, model, loader, criterion, metric_fns, collect_predictions=False, max_batches=0):
        results = super()._evaluate(model, loader, criterion, metric_fns, collect_predictions, max_batches=max_batches)
        ppl = math.exp(min(results['loss'], 100))
        results['perplexity'] = ppl
        return results

    def _train_one_epoch(self, model, loader, criterion, optimizer, logger, metric_fns, epoch):
        results = super()._train_one_epoch(model, loader, criterion, optimizer, logger, metric_fns, epoch)
        ppl = math.exp(min(results['loss'], 100))
        results['perplexity'] = ppl
        return results
