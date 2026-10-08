# Copyright (c) ModelScope Contributors. All rights reserved.
"""Skip ZeRO-1/2 optimizer steps whose global gradient norm is an outlier spike.

DeepSpeed's bf16 ``check_grad_overflow`` only skips non-finite gradients. Expert-RL runs also
produce rare finite spikes (global grad norm 1.5e4-1.1e8 against a ~0.1 baseline, clip 1.0):
clipping bounds the update, but the spike inflates Adam's second moment and damps the following
steps. With ``SWIFT_GRAD_SPIKE_SKIP_NORM=<threshold>`` the partitioned-gradient overflow check
also reports an overflow when the global norm exceeds the threshold, so every rank skips the
update through DeepSpeed's existing overflow path (grads zeroed, ``skipped_steps`` incremented).
"""
import os
from typing import Optional

import torch

from swift.utils import get_logger

logger = get_logger()
_ENV = 'SWIFT_GRAD_SPIKE_SKIP_NORM'
_PATCHED_ATTR = '_swift_grad_spike_guard'


def spike_threshold() -> Optional[float]:
    value = os.environ.get(_ENV, '').strip()
    if not value:
        return None
    threshold = float(value)
    if not threshold > 0:
        raise ValueError(f'{_ENV} must be a positive number, got {value!r}')
    return threshold


def global_partition_grad_norm(optimizer) -> torch.Tensor:
    """sqrt(sum of squared partitioned gradients) over the data-parallel group (one all-reduce)."""
    device = None
    squares = None
    for index in range(len(optimizer.bit16_groups)):
        for grad in optimizer.averaged_gradients[index] or []:
            if grad is None:
                continue
            value = grad.detach().float().pow(2).sum()
            squares = value if squares is None else squares + value
            device = grad.device
    if squares is None:
        squares = torch.zeros((), dtype=torch.float32, device=device or 'cuda')
    squares = squares.reshape(1)
    group = getattr(optimizer, 'dp_process_group', None)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(squares, op=torch.distributed.ReduceOp.SUM, group=group)
    return squares.sqrt()


def install_grad_spike_guard() -> bool:
    """Patch ``DeepSpeedZeroOptimizer`` once when ``SWIFT_GRAD_SPIKE_SKIP_NORM`` is set."""
    threshold = spike_threshold()
    if threshold is None:
        return False
    try:
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    except Exception as exc:  # noqa: BLE001 - DeepSpeed is optional
        logger.warning(f'{_ENV} is set but DeepSpeed ZeRO-1/2 is unavailable: {exc}')
        return False
    if getattr(DeepSpeedZeroOptimizer, _PATCHED_ATTR, False):
        return True
    original = DeepSpeedZeroOptimizer.has_overflow_partitioned_grads_serial

    def has_overflow_partitioned_grads_serial(self):
        overflow = original(self)
        limit = spike_threshold()
        if limit is None:
            return overflow
        norm = global_partition_grad_norm(self)
        self.swift_last_global_grad_norm = float(norm.item())
        spike = norm > limit
        if bool(spike.item()):
            self.swift_grad_spike_skips = getattr(self, 'swift_grad_spike_skips', 0) + 1
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.warning(f'[grad-spike] global grad norm {self.swift_last_global_grad_norm:.4g} > {limit:g}; '
                               f'skipping this optimizer step (spike skips so far: {self.swift_grad_spike_skips}).')
        return overflow | spike.to(device=overflow.device).reshape(overflow.shape)

    DeepSpeedZeroOptimizer.has_overflow_partitioned_grads_serial = has_overflow_partitioned_grads_serial
    setattr(DeepSpeedZeroOptimizer, _PATCHED_ATTR, True)
    logger.info(f'Gradient spike guard enabled: ZeRO-1/2 steps with global grad norm > {threshold:g} are skipped '
                '(requires bf16.check_grad_overflow or fp16).')
    return True
