"""NSP Runner — Next Sentence Prediction training loop."""

from contextlib import nullcontext

from experiment.runner import BaseRunner


class NSPRunner(BaseRunner):
    """
    Next Sentence Prediction runner.

    Batch format: (input_ids, attention_mask, token_type_ids, nsp_labels)
    Loss: CrossEntropyLoss (2-class on pooled output)
    Metrics: NSP accuracy
    """

    def setup_metrics(self):
        def nsp_accuracy(logits, targets):
            if logits is None or targets is None:
                return 0.0
            if hasattr(logits, 'detach'):
                return (logits.argmax(-1) == targets).float().sum().item()
            import numpy as np
            logits_np = logits.data if hasattr(logits, 'data') else logits
            targets_np = targets.data if hasattr(targets, 'data') else targets
            return float((logits_np.argmax(-1) == targets_np).sum())
        return {'nsp_accuracy': nsp_accuracy}

    def setup_criterion(self):
        if self.config.backend == 'pytorch':
            import torch.nn.functional as F
            return F.cross_entropy
        from python.optimization.losses import CrossEntropyLoss
        return CrossEntropyLoss()

    def train_step(self, model, batch, criterion, optimizer, is_accumulating=False):
        if self.config.backend == 'numpy':
            return self._np_train_step(model, batch, criterion, optimizer)

        import torch
        backend = self.backend
        grad_accum = self.config.gradient_accumulation_steps

        input_ids = batch[0].to(backend.device, non_blocking=True)
        attention_mask = batch[1].to(backend.device, non_blocking=True)
        token_type_ids = batch[2].to(backend.device, non_blocking=True)
        nsp_labels = batch[3].to(backend.device, non_blocking=True)

        ctx = model.no_sync() if (is_accumulating and hasattr(model, 'no_sync')) else nullcontext()

        with ctx:
            if backend._use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                    loss = criterion(logits, nsp_labels)
                    if grad_accum > 1:
                        loss = loss / grad_accum
                backend._scaler.scale(loss).backward()
            else:
                logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                loss = criterion(logits, nsp_labels)
                if grad_accum > 1:
                    loss = loss / grad_accum
                loss.backward()

        raw_loss = loss.detach() * grad_accum if grad_accum > 1 else loss.detach()
        return logits.detach(), raw_loss, input_ids.shape[0], nsp_labels

    def _np_train_step(self, model, batch, criterion, optimizer):
        from python.foundations.computational_graph import Tensor

        input_ids = batch[0]
        if hasattr(input_ids, 'numpy'): input_ids = input_ids.numpy()
        attention_mask = batch[1]
        if hasattr(attention_mask, 'numpy'): attention_mask = attention_mask.numpy()
        token_type_ids = batch[2]
        if hasattr(token_type_ids, 'numpy'): token_type_ids = token_type_ids.numpy()
        nsp_labels = batch[3]
        if hasattr(nsp_labels, 'numpy'): nsp_labels = nsp_labels.numpy()

        x = Tensor(input_ids, requires_grad=True)
        tt = Tensor(token_type_ids)
        am = Tensor(attention_mask)

        logits = model(x, token_type_ids=tt, attention_mask=am)
        loss = criterion(logits, Tensor(nsp_labels), reduction='mean')
        loss.backward()

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return logits, batch_loss, len(input_ids), nsp_labels

    def eval_step(self, model, batch, criterion):
        if self.config.backend == 'numpy':
            return self._np_eval_step(model, batch, criterion)

        import torch
        backend = self.backend
        input_ids = batch[0].to(backend.device, non_blocking=True)
        attention_mask = batch[1].to(backend.device, non_blocking=True)
        token_type_ids = batch[2].to(backend.device, non_blocking=True)
        nsp_labels = batch[3].to(backend.device, non_blocking=True)

        if backend._use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                loss = criterion(logits, nsp_labels)
        else:
            logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss = criterion(logits, nsp_labels)

        return logits.detach(), loss.detach(), input_ids.shape[0], nsp_labels

    def _np_eval_step(self, model, batch, criterion):
        from python.foundations.computational_graph import Tensor

        input_ids = batch[0]
        if hasattr(input_ids, 'numpy'): input_ids = input_ids.numpy()
        attention_mask = batch[1]
        if hasattr(attention_mask, 'numpy'): attention_mask = attention_mask.numpy()
        token_type_ids = batch[2]
        if hasattr(token_type_ids, 'numpy'): token_type_ids = token_type_ids.numpy()
        nsp_labels = batch[3]
        if hasattr(nsp_labels, 'numpy'): nsp_labels = nsp_labels.numpy()

        x = Tensor(input_ids, requires_grad=False)
        tt = Tensor(token_type_ids)
        am = Tensor(attention_mask)

        logits = model(x, token_type_ids=tt, attention_mask=am)
        loss = criterion(logits, Tensor(nsp_labels), reduction='mean')

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return logits, batch_loss, len(input_ids), nsp_labels

    def format_metric(self, name, value):
        if name == 'loss':
            return f"{value:.4f}"
        if name == 'nsp_accuracy':
            return f"{value*100:.2f}%"
        return f"{value:.4f}"
