"""TokenClassificationRunner — Token-level classification (POS tagging, NER)."""

from contextlib import nullcontext

from experiment.runner import BaseRunner


class TokenClassificationRunner(BaseRunner):
    """
    Token classification runner.

    Batch format: (input_ids, attention_mask, labels)
    Loss: CrossEntropyLoss(ignore_index=-100) on [B*T, num_tags]
    Metrics: token-level accuracy (ignoring -100)
    """

    def setup_metrics(self):
        def _to_np(x):
            import numpy as np
            if isinstance(x, np.ndarray):
                return x
            if hasattr(x, 'detach'):
                return x.detach().cpu().numpy()
            if hasattr(x, 'data') and isinstance(x.data, np.ndarray):
                return x.data
            return np.asarray(x)

        def token_accuracy(logits, targets):
            """Return accuracy * batch_size so runner's /total_samples gives correct avg."""
            import numpy as np
            if logits is None or targets is None:
                return 0.0
            logits_np = _to_np(logits)
            targets_np = _to_np(targets)
            B = logits_np.shape[0]
            mask = targets_np != -100
            n_valid = mask.sum()
            if n_valid == 0:
                return 0.0
            preds = logits_np.argmax(-1)
            acc = float((preds[mask] == targets_np[mask]).sum()) / float(n_valid)
            return acc * B
        return {'token_accuracy': token_accuracy}

    def setup_criterion(self):
        if self.config.backend == 'pytorch':
            import torch.nn.functional as F
            def token_cls_criterion(logits, targets):
                B, T, C = logits.shape
                return F.cross_entropy(
                    logits.reshape(B * T, C), targets.reshape(B * T),
                    ignore_index=-100,
                )
            return token_cls_criterion
        from python.optimization.losses import CrossEntropyLoss
        ce = CrossEntropyLoss()
        def np_token_cls_criterion(logits, targets, reduction='mean'):
            return ce(logits, targets, ignore_index=-100, reduction=reduction)
        return np_token_cls_criterion

    def train_step(self, model, batch, criterion, optimizer, is_accumulating=False):
        if self.config.backend == 'numpy':
            return self._np_train_step(model, batch, criterion, optimizer)

        import torch
        backend = self.backend
        grad_accum = self.config.gradient_accumulation_steps

        input_ids = batch[0].to(backend.device, non_blocking=True)
        attention_mask = batch[1].to(backend.device, non_blocking=True)
        labels = batch[2].to(backend.device, non_blocking=True)

        ctx = model.no_sync() if (is_accumulating and hasattr(model, 'no_sync')) else nullcontext()

        with ctx:
            if backend._use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
                    if grad_accum > 1:
                        loss = loss / grad_accum
                backend._scaler.scale(loss).backward()
            else:
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                if grad_accum > 1:
                    loss = loss / grad_accum
                loss.backward()

        raw_loss = loss.detach() * grad_accum if grad_accum > 1 else loss.detach()
        return logits.detach(), raw_loss, input_ids.shape[0], labels

    def _np_train_step(self, model, batch, criterion, optimizer):
        from python.foundations.computational_graph import Tensor

        input_ids = batch[0]
        if hasattr(input_ids, 'numpy'): input_ids = input_ids.numpy()
        attention_mask = batch[1]
        if hasattr(attention_mask, 'numpy'): attention_mask = attention_mask.numpy()
        labels = batch[2]
        if hasattr(labels, 'numpy'): labels = labels.numpy()

        x = Tensor(input_ids, requires_grad=True)
        am = Tensor(attention_mask)

        logits = model(x, attention_mask=am)

        B, T = labels.shape
        logits_flat = logits.reshape(B * T, -1)
        labels_flat = Tensor(labels.reshape(B * T))

        loss = criterion(logits_flat, labels_flat, reduction='mean')
        loss.backward()

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return logits, batch_loss, B, labels

    def eval_step(self, model, batch, criterion):
        if self.config.backend == 'numpy':
            return self._np_eval_step(model, batch, criterion)

        import torch
        backend = self.backend
        input_ids = batch[0].to(backend.device, non_blocking=True)
        attention_mask = batch[1].to(backend.device, non_blocking=True)
        labels = batch[2].to(backend.device, non_blocking=True)

        if backend._use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
        else:
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        return logits.detach(), loss.detach(), input_ids.shape[0], labels

    def _np_eval_step(self, model, batch, criterion):
        from python.foundations.computational_graph import Tensor

        input_ids = batch[0]
        if hasattr(input_ids, 'numpy'): input_ids = input_ids.numpy()
        attention_mask = batch[1]
        if hasattr(attention_mask, 'numpy'): attention_mask = attention_mask.numpy()
        labels = batch[2]
        if hasattr(labels, 'numpy'): labels = labels.numpy()

        x = Tensor(input_ids, requires_grad=False)
        am = Tensor(attention_mask)

        logits = model(x, attention_mask=am)

        B, T = labels.shape
        logits_flat = logits.reshape(B * T, -1)
        labels_flat = Tensor(labels.reshape(B * T))

        loss = criterion(logits_flat, labels_flat, reduction='mean')

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return logits, batch_loss, B, labels

    def format_metric(self, name, value):
        if name == 'loss':
            return f"{value:.4f}"
        if name == 'token_accuracy':
            return f"{value*100:.2f}%"
        return f"{value:.4f}"
