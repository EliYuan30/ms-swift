# Copyright (c) ModelScope Contributors. All rights reserved.
"""Call-boundary fixes for FlashAttention-4 (``flash_attn.cute``) under transformers.

transformers passes ``max_seqlen_q/k`` to the varlen kernel as 0-dim tensors: ``_unpad_input``
returns ``seqlens.max()`` and the Qwen3.5 vision tower computes ``(cu[1:] - cu[:-1]).max()``.
FlashAttention-2's C++ binding converts them to ``int``. With flash-attn-4 4.0.0b19 the tensor
values made the backward recompile on every call (Qwen3.5-9B, one B200: ~159 s per
forward/backward instead of ~0.2 s). Converting them to Python ``int`` removes the recompiles.
On captured Qwen3.5 attention inputs FA4's per-call error against fp32 matches FA2's.
"""
import functools
import inspect
import os

import torch

from swift.utils import get_logger

logger = get_logger()

_SCALAR_ARGUMENTS = ('max_seqlen_q', 'max_seqlen_k', 'min_seqlen_k')
# Defensive: the Qwen3.5 vision tower passes value views sliced out of the fused qkv projection.
# b19 handled such row strides correctly on GPU, and a dense copy is cheap next to attention.
_TENSOR_ARGUMENTS = ('q', 'k', 'v', 'qv', 'cu_seqlens_q', 'cu_seqlens_k')
_PATCHED_ATTR = '_swift_scalar_seqlen_patch'


def vision_attn_impl_for_fa4() -> str:
    """Attention implementation for vision/audio towers when the LLM uses FA4.

    flash-attn-4 4.0.0b19 cannot compile the backward for the Qwen3.5 ViT head dim 72 (ICE in
    ``flash_bwd_preprocess``), and the ViT stays in the autograd graph during RL training even with
    frozen weights, so the towers default to FlashAttention-2. ``SWIFT_FA4_VISION_ATTN_IMPL``
    overrides it (e.g. ``flash_attention_4`` for inference-only use).
    """
    return os.environ.get('SWIFT_FA4_VISION_ATTN_IMPL', 'flash_attention_2')


def _as_python_int(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f'expected a scalar sequence length, got shape {tuple(value.shape)}')
        return int(value.item())
    return value


def _with_scalar_seqlens(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        for name in _SCALAR_ARGUMENTS:
            if name in bound.arguments:
                bound.arguments[name] = _as_python_int(bound.arguments[name])
        for name in _TENSOR_ARGUMENTS:
            value = bound.arguments.get(name)
            if isinstance(value, torch.Tensor) and not value.is_contiguous():
                bound.arguments[name] = value.contiguous()
        return function(*bound.args, **bound.kwargs)

    setattr(wrapper, _PATCHED_ATTR, True)
    return wrapper


def patch_flash_attn_4_scalar_seqlens() -> bool:
    """Wrap ``flash_attn.cute.flash_attn_varlen_func`` once; returns whether FA4 is patched.

    transformers imports the kernels from ``flash_attn.cute`` lazily at the first attention call
    and inspects their signatures; ``functools.wraps`` keeps that signature, so the requested
    kwargs are unchanged.
    """
    try:
        import flash_attn.cute as fa4
    except Exception as exc:  # noqa: BLE001 - FA4 is optional
        logger.warning(f'flash_attention_4 requested but flash_attn.cute is unavailable: {exc}')
        return False
    function = getattr(fa4, 'flash_attn_varlen_func', None)
    if function is None:
        return False
    if not getattr(function, _PATCHED_ATTR, False):
        fa4.flash_attn_varlen_func = _with_scalar_seqlens(function)
        logger.info('flash_attention_4: varlen max_seqlen arguments are passed as Python ints.')
    return True
