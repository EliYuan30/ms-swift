# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in gradient attribution for diagnosing finite gradient-norm spikes.

``SWIFT_GRAD_ATTRIBUTION=1`` registers a tensor hook on every trainable parameter. Each hook
adds the squared norm of the incoming local (pre-reduction) gradient to a per-parameter GPU
buffer; after every micro-step the buffer is snapshotted with light batch metadata. At each
gradient-accumulation boundary one JSON line per rank is appended to
``<dir>/rank<R>.jsonl`` with the per-micro local norms, the largest parameters and per-module
buckets, so a spike can be localized to a micro-batch (samples) and a module type.

When a micro-step's local norm exceeds ``SWIFT_GRAD_ATTRIBUTION_DUMP_NORM`` (default 100), the
micro-batch tensors (model inputs and loss batch, on CPU) are saved for offline replay, up to
``SWIFT_GRAD_ATTRIBUTION_MAX_DUMPS`` (default 3) per rank. The hooks only read gradients; the
training math is unchanged. The directory defaults to ``<output_dir>/grad_attribution`` and can
be overridden with ``SWIFT_GRAD_ATTRIBUTION_DIR``.
"""
import json
import math
import os
import re
from dataclasses import fields, is_dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import torch

from swift.utils import get_logger

logger = get_logger()
_ENV = 'SWIFT_GRAD_ATTRIBUTION'
_LAYER_RE = re.compile(r'\.\d+\.')


def attribution_enabled() -> bool:
    return os.environ.get(_ENV, '0') == '1'


def bucket_name(name: str) -> str:
    """Collapse layer indices and the trailing weight/bias so modules aggregate across layers."""
    name = name.replace('_checkpoint_wrapped_module.', '').replace('base_model.model.', '')
    name = _LAYER_RE.sub('.*.', name)
    for suffix in ('.weight', '.bias'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def layer_name(name: str) -> str:
    """'L<idx>' for language-model decoder layers, 'V<idx>' for vision blocks, else the top module."""
    match = re.search(r'layers\.(\d+)\.', name)
    if match and 'visual' not in name:
        return f'L{int(match.group(1)):02d}'
    match = re.search(r'blocks\.(\d+)\.', name)
    if match:
        return f'V{int(match.group(1)):02d}'
    return bucket_name(name)


def _row_stats(values: Optional[torch.Tensor], mask: Optional[torch.Tensor], reducer: str) -> Optional[List[float]]:
    if values is None or mask is None or values.dim() != 2 or values.shape != mask.shape:
        return None
    mask = mask.bool()
    fill = float('inf') if reducer == 'min' else float('-inf')
    filled = values.detach().float().masked_fill(~mask, fill)
    out = filled.min(dim=-1).values if reducer == 'min' else filled.max(dim=-1).values
    return [round(float(v), 4) for v in out.cpu()]


def micro_batch_metadata(inputs: Any) -> Dict[str, Any]:
    """Small, JSON-serializable description of one micro-batch (never raises)."""
    meta: Dict[str, Any] = {}
    try:
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        if not isinstance(inputs, dict):
            return meta
        model_inputs = inputs.get('model_inputs') or {}
        batch = inputs.get('grpo_batch')
        for key, value in model_inputs.items():
            if isinstance(value, torch.Tensor):
                meta.setdefault('shapes', {})[key] = list(value.shape)
        for key in ('image_grid_thw', 'video_grid_thw'):
            value = model_inputs.get(key)
            if isinstance(value, torch.Tensor):
                meta[key] = value.tolist()
        if batch is not None:
            mask = getattr(batch, 'completion_mask', None)
            if isinstance(mask, torch.Tensor):
                meta['completion_tokens'] = mask.sum(-1).tolist()
            for name in ('seq_lengths', 'truncated_mask', 'sequence_loss_weights'):
                value = getattr(batch, name, None)
                if isinstance(value, torch.Tensor):
                    meta[name] = [round(float(v), 4) for v in value.detach().float().cpu()]
            advantages = getattr(batch, 'advantages', None)
            if isinstance(advantages, torch.Tensor):
                adv = advantages.detach().float()
                meta['advantage'] = [round(float(v), 4) for v in (adv[:, 0] if adv.dim() == 2 else adv).cpu()]
            meta['rollout_logp_min'] = _row_stats(getattr(batch, 'rollout_per_token_logps', None), mask, 'min')
            meta['old_logp_min'] = _row_stats(getattr(batch, 'old_per_token_logps', None), mask, 'min')
            meta['ref_logp_min'] = _row_stats(getattr(batch, 'ref_per_token_logps', None), mask, 'min')
        samples = inputs.get('_origin_data') or []
        rows = []
        for sample in samples:
            infos = getattr(sample, 'rollout_infos', None) or {}
            extra = getattr(sample, 'extra', None) or {}
            rows.append({
                'prompt_id': str(getattr(sample, 'prompt_id', ''))[:80],
                'request_id': str(getattr(sample, 'request_id', '') or infos.get('request_id', ''))[:80],
                'qid': str(extra.get('qid', extra.get('id', '')))[:80],
                'source': str(extra.get('source', extra.get('data_source', '')))[:40],
                'n_images': len(getattr(sample, 'images', None) or []),
                'n_videos': len(getattr(sample, 'videos', None) or []),
                'n_messages': len(getattr(sample, 'messages', None) or []),
                'turn': infos.get('training_turn_index', infos.get('turn_index')),
                'turn_count': infos.get('training_turn_count'),
            })
        if rows:
            meta['samples'] = rows
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break training
        meta['metadata_error'] = repr(exc)[:200]
    return meta


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_cpu(getattr(value, f.name)) for f in fields(value)}
    return value


class GradAttribution:

    def __init__(self, model: torch.nn.Module, out_dir: str, rank: int, top_k: int = 8):
        self.out_dir = out_dir
        self.rank = rank
        self.top_k = top_k
        self.dump_norm = float(os.environ.get('SWIFT_GRAD_ATTRIBUTION_DUMP_NORM', '100'))
        self.max_dumps = int(os.environ.get('SWIFT_GRAD_ATTRIBUTION_MAX_DUMPS', '3'))
        self.dumps = 0
        self.names: List[str] = []
        self.buckets: List[str] = []
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not params:
            raise ValueError('GradAttribution: the model has no trainable parameters.')
        device = params[0][1].device
        self.current = torch.zeros(len(params), dtype=torch.float32, device=device)
        self.handles = []
        for index, (name, param) in enumerate(params):
            self.names.append(name)
            self.buckets.append(bucket_name(name))
            self.handles.append(param.register_hook(partial(self._hook, index)))
        self.layer_ids = sorted({layer_name(n) for n in self.names})
        self._layer_index = torch.tensor([self.layer_ids.index(layer_name(n)) for n in self.names], dtype=torch.long)
        self.bucket_ids = sorted(set(self.buckets))
        self._bucket_index = torch.tensor([self.bucket_ids.index(b) for b in self.buckets], dtype=torch.long)
        self.micros: List[Dict[str, Any]] = []
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f'rank{rank}.jsonl')

    def _hook(self, index: int, grad: torch.Tensor):
        if self.current.device != grad.device:
            self.current = self.current.to(grad.device)
        # bf16 norms accumulate in fp32 internally; avoids a full fp32 copy of large gradients.
        norm = torch.linalg.vector_norm(grad.detach()).float()
        self.current[index] += norm * norm
        return None

    def end_micro(self, inputs: Any, global_step: int, boundary: bool, guard_norm: Optional[float]) -> None:
        squares = self.current.clone()
        self.current.zero_()
        local = float(squares.sum().sqrt().item())
        micro = {'local_norm': local, 'squares': squares, 'meta': micro_batch_metadata(inputs)}
        if (local > self.dump_norm or not math.isfinite(local)) and self.dumps < self.max_dumps:
            micro['dump'] = self._dump(inputs, global_step, len(self.micros), local, squares)
        self.micros.append(micro)
        if boundary:
            self.flush(global_step, guard_norm)

    def _top(self, squares: torch.Tensor) -> List[List[Any]]:
        k = min(self.top_k, squares.numel())
        values, indices = squares.topk(k)
        return [[self.names[i], round(float(v.sqrt()), 6)] for v, i in zip(values.cpu(), indices.cpu().tolist())]

    def _bucket_norms(self, squares: torch.Tensor) -> List[List[Any]]:
        sums = torch.zeros(len(self.bucket_ids), dtype=torch.float32)
        sums.index_add_(0, self._bucket_index, squares.cpu())
        order = sums.argsort(descending=True)[:self.top_k].tolist()
        return [[self.bucket_ids[i], round(float(sums[i].sqrt()), 6)] for i in order]

    def _layer_norms(self, squares: torch.Tensor) -> List[List[Any]]:
        sums = torch.zeros(len(self.layer_ids), dtype=torch.float32)
        sums.index_add_(0, self._layer_index, squares.cpu())
        return [[self.layer_ids[i], round(float(sums[i].sqrt()), 6)] for i in range(len(self.layer_ids))]

    def flush(self, global_step: int, guard_norm: Optional[float]) -> None:
        if not self.micros:
            return
        total = torch.stack([m['squares'] for m in self.micros]).sum(0)
        worst = max(range(len(self.micros)), key=lambda i: self.micros[i]['local_norm'])
        record = {
            'global_step': global_step,
            'rank': self.rank,
            'guard_global_norm': guard_norm,
            'local_step_norm': round(float(total.sum().sqrt()), 6),
            'micro_local_norms': [round(m['local_norm'], 6) for m in self.micros],
            'worst_micro': worst,
            'worst_top_params': self._top(self.micros[worst]['squares']),
            'worst_buckets': self._bucket_norms(self.micros[worst]['squares']),
            'worst_meta': self.micros[worst]['meta'],
            'worst_layers': self._layer_norms(self.micros[worst]['squares']),
            'step_buckets': self._bucket_norms(total),
            'dumps': [m['dump'] for m in self.micros if m.get('dump')],
        }
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, default=str) + '\n')
        if record['local_step_norm'] > self.dump_norm:
            logger.warning(f'[grad-attribution] rank {self.rank} step {global_step}: local norm '
                           f'{record["local_step_norm"]:.4g}, worst micro {worst} '
                           f'({record["micro_local_norms"][worst]:.4g}), top {record["worst_top_params"][:3]}')
        self.micros = []

    def _dump(self, inputs: Any, global_step: int, micro: int, local: float, squares: torch.Tensor) -> Optional[str]:
        path = os.path.join(self.out_dir, f'spike_step{global_step}_rank{self.rank}_micro{micro}.pt')
        try:
            if isinstance(inputs, list) and len(inputs) == 1:
                inputs = inputs[0]
            payload = {
                'global_step': global_step,
                'micro': micro,
                'rank': self.rank,
                'local_norm': local,
                'top_params': self._top(squares),
                'model_inputs': _to_cpu(inputs.get('model_inputs')),
                'grpo_batch': _to_cpu(inputs.get('grpo_batch')),
                'meta': micro_batch_metadata(inputs),
            }
            torch.save(payload, path)
            self.dumps += 1
            logger.warning(f'[grad-attribution] rank {self.rank} step {global_step} micro {micro}: local norm '
                           f'{local:.4g} > {self.dump_norm:g}; saved micro-batch to {path}')
            return path
        except Exception as exc:  # noqa: BLE001
            logger.warning(f'[grad-attribution] failed to save micro-batch: {exc!r}')
            return None


def install_grad_attribution(trainer) -> Optional[GradAttribution]:
    if not attribution_enabled():
        return None
    model = trainer.accelerator.unwrap_model(trainer.model) if hasattr(trainer, 'accelerator') else trainer.model
    out_dir = os.environ.get('SWIFT_GRAD_ATTRIBUTION_DIR') or os.path.join(trainer.args.output_dir,
                                                                             'grad_attribution')
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    attribution = GradAttribution(model, out_dir, rank)
    logger.info(f'Gradient attribution enabled for {len(attribution.names)} trainable tensors -> {out_dir}')
    return attribution
