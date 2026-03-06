"""BertPreTrainRunner — Joint MLM + NSP training loop."""

import math
from contextlib import nullcontext

from experiment.runner import BaseRunner


class BertPreTrainRunner(BaseRunner):
    """
    BERT Pre-Training runner (joint MLM + NSP).

    Batch format: dict {input_ids, attention_mask, token_type_ids, mlm_labels, nsp_labels}
    Model: BertForPreTraining -> (mlm_logits, nsp_logits)
    Loss: mlm_loss + nsp_loss
    Metrics: MLM accuracy, MLM perplexity, NSP accuracy
    """

    def setup_metrics(self):
        def _to_np(x):
            import numpy as np
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x
            if hasattr(x, 'detach'):
                return x.detach().cpu().numpy()
            if hasattr(x, 'data') and isinstance(x.data, np.ndarray):
                return x.data
            return np.asarray(x)

        def mlm_accuracy(logits_tuple, targets_tuple):
            import numpy as np
            if logits_tuple is None or targets_tuple is None:
                return 0.0
            mlm_logits = logits_tuple[0] if isinstance(logits_tuple, tuple) else logits_tuple
            mlm_labels = targets_tuple[0] if isinstance(targets_tuple, tuple) else targets_tuple
            mlm_logits = _to_np(mlm_logits)
            mlm_labels = _to_np(mlm_labels)
            if mlm_logits is None or mlm_labels is None:
                return 0.0
            B = mlm_logits.shape[0]
            mask = mlm_labels != -100
            n_valid = mask.sum()
            if n_valid == 0:
                return 0.0
            preds = mlm_logits.argmax(-1)
            acc = float((preds[mask] == mlm_labels[mask]).sum()) / float(n_valid)
            return acc * B

        def nsp_accuracy(logits_tuple, targets_tuple):
            import numpy as np
            if logits_tuple is None or targets_tuple is None:
                return 0.0
            if not isinstance(logits_tuple, tuple) or len(logits_tuple) < 2:
                return 0.0
            nsp_logits = _to_np(logits_tuple[1])
            nsp_labels = _to_np(targets_tuple[1]) if isinstance(targets_tuple, tuple) and len(targets_tuple) > 1 else None
            if nsp_logits is None or nsp_labels is None:
                return 0.0
            B = nsp_logits.shape[0]
            return float((nsp_logits.argmax(-1) == nsp_labels).sum())

        return {'mlm_accuracy': mlm_accuracy, 'nsp_accuracy': nsp_accuracy}

    def setup_criterion(self):
        if self.config.backend == 'pytorch':
            import torch.nn.functional as F
            def pretrain_criterion(logits_tuple, targets_tuple):
                mlm_logits, nsp_logits = logits_tuple
                mlm_labels, nsp_labels = targets_tuple
                B, T, V = mlm_logits.shape
                mlm_loss = F.cross_entropy(
                    mlm_logits.reshape(B * T, V), mlm_labels.reshape(B * T),
                    ignore_index=-100,
                )
                nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
                return mlm_loss + nsp_loss
            return pretrain_criterion
        from python.optimization.losses import CrossEntropyLoss
        ce = CrossEntropyLoss()
        def np_pretrain_criterion(logits_tuple, targets_tuple, reduction='mean'):
            from python.foundations.computational_graph import Tensor
            mlm_logits, nsp_logits = logits_tuple
            mlm_labels, nsp_labels = targets_tuple
            B, T = mlm_labels.shape if hasattr(mlm_labels, 'shape') else (mlm_labels.data.shape[0], mlm_labels.data.shape[1])
            mlm_logits_flat = mlm_logits.reshape(B * T, -1)
            mlm_labels_flat = Tensor(mlm_labels.reshape(B * T)) if not hasattr(mlm_labels, 'reshape') else mlm_labels.reshape(B * T)
            mlm_loss = ce(mlm_logits_flat, mlm_labels_flat, ignore_index=-100, reduction=reduction)
            nsp_loss = ce(nsp_logits, nsp_labels, reduction=reduction)
            return mlm_loss + nsp_loss
        return np_pretrain_criterion

    def train_step(self, model, batch, criterion, optimizer, is_accumulating=False):
        if self.config.backend == 'numpy':
            return self._np_train_step(model, batch, criterion, optimizer)

        import torch
        backend = self.backend
        grad_accum = self.config.gradient_accumulation_steps

        input_ids = batch['input_ids'].to(backend.device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(backend.device, non_blocking=True)
        token_type_ids = batch['token_type_ids'].to(backend.device, non_blocking=True)
        mlm_labels = batch['mlm_labels'].to(backend.device, non_blocking=True)
        nsp_labels = batch['nsp_labels'].to(backend.device, non_blocking=True)

        ctx = model.no_sync() if (is_accumulating and hasattr(model, 'no_sync')) else nullcontext()

        with ctx:
            if backend._use_amp:
                with torch.amp.autocast('cuda'):
                    mlm_logits, nsp_logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                    loss = criterion((mlm_logits, nsp_logits), (mlm_labels, nsp_labels))
                    if grad_accum > 1:
                        loss = loss / grad_accum
                backend._scaler.scale(loss).backward()
            else:
                mlm_logits, nsp_logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                loss = criterion((mlm_logits, nsp_logits), (mlm_labels, nsp_labels))
                if grad_accum > 1:
                    loss = loss / grad_accum
                loss.backward()

        raw_loss = loss.detach() * grad_accum if grad_accum > 1 else loss.detach()
        return (mlm_logits.detach(), nsp_logits.detach()), raw_loss, input_ids.shape[0], (mlm_labels, nsp_labels)

    def _unpack_np_batch(self, batch):
        """Unpack batch from either dict or tuple format."""
        if isinstance(batch, dict):
            return batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'), batch.get('mlm_labels'), batch.get('nsp_labels')
        # Tuple format from NpLoader: (input_ids, attention_mask, mlm_labels, token_type_ids)
        input_ids, attention_mask, mlm_labels, token_type_ids = batch[0], batch[1], batch[2], batch[3]
        nsp_labels = batch[4] if len(batch) > 4 else None
        return input_ids, attention_mask, token_type_ids, mlm_labels, nsp_labels

    def _np_train_step(self, model, batch, criterion, optimizer):
        from python.foundations.computational_graph import Tensor
        import numpy as np

        input_ids, attention_mask, token_type_ids, mlm_labels, nsp_labels = self._unpack_np_batch(batch)
        for arr_name in ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_labels', 'nsp_labels']:
            arr = locals()[arr_name]
            if arr is not None and hasattr(arr, 'numpy'):
                locals()[arr_name] = arr.numpy()

        x = Tensor(input_ids, requires_grad=True)
        tt = Tensor(token_type_ids) if token_type_ids is not None else None
        am = Tensor(attention_mask)

        # For MLM-only fallback (no NSP), just compute MLM
        if nsp_labels is None:
            mlm_logits = model(x, token_type_ids=tt, attention_mask=am)
            if isinstance(mlm_logits, tuple):
                mlm_logits, nsp_logits = mlm_logits
            else:
                nsp_logits = None
            if nsp_logits is not None:
                nsp_labels_t = Tensor(np.zeros(len(input_ids), dtype=np.int64))
                loss = criterion((mlm_logits, nsp_logits), (Tensor(mlm_labels), nsp_labels_t), reduction='mean')
            else:
                B, T = mlm_labels.shape
                logits_flat = mlm_logits.reshape(B * T, -1)
                labels_flat = Tensor(mlm_labels.reshape(B * T))
                from python.optimization.losses import CrossEntropyLoss
                ce = CrossEntropyLoss()
                loss = ce(logits_flat, labels_flat, ignore_index=-100, reduction='mean')
        else:
            mlm_logits, nsp_logits = model(x, token_type_ids=tt, attention_mask=am)
            loss = criterion((mlm_logits, nsp_logits), (Tensor(mlm_labels), Tensor(nsp_labels)), reduction='mean')

        loss.backward()
        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return (mlm_logits, nsp_logits if nsp_labels is not None else mlm_logits), batch_loss, len(input_ids), (mlm_labels, nsp_labels if nsp_labels is not None else mlm_labels)

    def eval_step(self, model, batch, criterion):
        if self.config.backend == 'numpy':
            return self._np_eval_step(model, batch, criterion)

        import torch
        backend = self.backend
        input_ids = batch['input_ids'].to(backend.device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(backend.device, non_blocking=True)
        token_type_ids = batch['token_type_ids'].to(backend.device, non_blocking=True)
        mlm_labels = batch['mlm_labels'].to(backend.device, non_blocking=True)
        nsp_labels = batch['nsp_labels'].to(backend.device, non_blocking=True)

        if backend._use_amp:
            with torch.amp.autocast('cuda'):
                mlm_logits, nsp_logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                loss = criterion((mlm_logits, nsp_logits), (mlm_labels, nsp_labels))
        else:
            mlm_logits, nsp_logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss = criterion((mlm_logits, nsp_logits), (mlm_labels, nsp_labels))

        return (mlm_logits.detach(), nsp_logits.detach()), loss.detach(), input_ids.shape[0], (mlm_labels, nsp_labels)

    def _np_eval_step(self, model, batch, criterion):
        from python.foundations.computational_graph import Tensor
        import numpy as np

        input_ids, attention_mask, token_type_ids, mlm_labels, nsp_labels = self._unpack_np_batch(batch)

        x = Tensor(input_ids, requires_grad=False)
        tt = Tensor(token_type_ids) if token_type_ids is not None else None
        am = Tensor(attention_mask)

        output = model(x, token_type_ids=tt, attention_mask=am)
        if isinstance(output, tuple):
            mlm_logits, nsp_logits = output
        else:
            mlm_logits = output
            nsp_logits = None

        if nsp_labels is not None and nsp_logits is not None:
            loss = criterion((mlm_logits, nsp_logits), (Tensor(mlm_labels), Tensor(nsp_labels)), reduction='mean')
        else:
            B, T = mlm_labels.shape
            from python.optimization.losses import CrossEntropyLoss
            ce = CrossEntropyLoss()
            loss = ce(mlm_logits.reshape(B * T, -1), Tensor(mlm_labels.reshape(B * T)), ignore_index=-100, reduction='mean')

        batch_loss = loss.data.item() if loss.data.ndim == 0 else float(loss.data.mean())
        return (mlm_logits, nsp_logits), batch_loss, len(input_ids), (mlm_labels, nsp_labels)

    def format_metric(self, name, value):
        if name == 'loss':
            return f"{value:.4f}"
        if name == 'perplexity':
            return f"{value:.2f}"
        if 'accuracy' in name:
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
