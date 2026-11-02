"""Fused-only GDN dispatch tests: the wrapper never falls back.

``ascendc_chunk_gated_delta_rule`` targets the 26.10.0 fused
``npu_chunk_gated_delta_rule_fwd/bwd`` pair exclusively. These tests
replace the fused autograd Function with recorders (nothing launches on
device) and lock down the wrapper's contract:

* the happy path makes exactly one ``ChunkGatedDeltaRuleFusedFunction``
  call with the 14-arg fusion signature and never touches the composed
  ``ChunkGatedDeltaRuleFunction``;
* there is no escape hatch and no probe: no dispatch branch exists to
  exercise;
* shapes outside the fused envelope (V=256, chunk_size=128) fail loudly
  inside the fused adapter's own validation -- no fallback;
* ``save_backward_intermediates`` (a composed-chain-only option) is
  rejected up front.

These tests import the vendored adapter (which needs ``fla_npu``), so they
run on an NPU install; the importorskip guards make them a no-op elsewhere.
"""

import pytest

pytest.importorskip("torch_npu")
pytest.importorskip("fla_npu")

import torch  # noqa: E402

import torchturbo.ops.providers.npu._fla_npu_gdn as gdn_pkg  # noqa: E402


def _fake_tensors(batch=1, tokens=1057, key_heads=16, value_heads=48, value_dim=128):
    """Time-first inputs in the wrapper's fla-like calling convention.

    Head counts default to Qwen3.8-Flash-Next's linear-attention shapes
    (16 key : 48 value heads, K=V=128) so the happy-path dispatch is locked
    down on the production configuration, not just a toy one.
    """
    q = torch.randn(batch, tokens, key_heads, 128)
    k = torch.randn(batch, tokens, key_heads, 128)
    v = torch.randn(batch, tokens, value_heads, value_dim)
    g = -torch.rand(batch, tokens, value_heads)
    beta = torch.rand(batch, tokens, value_heads)
    return q, k, v, g, beta


class _Recorder:
    """Mimics a torch.autograd.Function: ``.apply(...)`` records the call."""

    def __init__(self):
        self.calls = []

    def apply(self, *args, **kwargs):  # noqa: D102 (stands in for Function)
        self.calls.append((args, kwargs))
        return torch.zeros(1), None


def _run_wrapper(monkeypatch, tensors, **extra):
    fused_rec = _Recorder()
    composed_rec = _Recorder()
    monkeypatch.setattr(gdn_pkg, "ChunkGatedDeltaRuleFusedFunction", fused_rec)
    monkeypatch.setattr(gdn_pkg, "ChunkGatedDeltaRuleFunction", composed_rec)
    q, k, v, g, beta = tensors
    o, final_state = gdn_pkg.ascendc_chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        use_qk_l2norm_in_kernel=False,
        chunk_size=64,
        **extra,
    )
    return fused_rec, composed_rec, o, final_state


def test_happy_path_is_fusion_only(monkeypatch):
    """Exactly one fused call with the 14-arg signature, no composed call."""
    fused, composed, *_ = _run_wrapper(monkeypatch, _fake_tensors())
    assert len(fused.calls) == 1
    assert len(composed.calls) == 0
    args = fused.calls[0][0]
    assert len(args) == 14  # composed chain passes 15 (save_intermediates)
    # q/k/v were transposed to head-first [B,H,T,D] on the way in
    assert tuple(args[0].shape) == (1, 16, 1057, 128)
    assert tuple(args[2].shape) == (1, 48, 1057, 128)
    assert args[5] == pytest.approx(128 ** -0.5)  # scale defaults to dk**-0.5
    assert args[13] == 64  # chunk_size


def test_outside_envelope_raises():
    """V=256 is inside the fused fwd's own envelope but outside the pair's
    (K=V=128). Nothing is patched: the real adapter's validation must
    reject it before any kernel launch -- there is no composed-chain
    fallback to catch it."""
    q = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.bfloat16)
    k = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.bfloat16)
    v = torch.randn(1, 64, 2, 256, device="npu", dtype=torch.bfloat16)
    g = -torch.rand(1, 64, 2, device="npu", dtype=torch.float32)
    beta = torch.rand(1, 64, 2, device="npu", dtype=torch.float32)
    with pytest.raises(ValueError, match="K=V=128"):
        gdn_pkg.ascendc_chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            use_qk_l2norm_in_kernel=False,
            chunk_size=64,
        )


def test_no_fallback_on_any_failure(monkeypatch):
    """No probe/escape-hatch branches exist, so even a fused adapter that
    raises propagates the error; the composed chain is never dispatched."""

    class _Boom:
        @staticmethod
        def apply(*args, **kwargs):
            raise RuntimeError("fused kernel missing on this install")

    monkeypatch.setattr(gdn_pkg, "ChunkGatedDeltaRuleFusedFunction", _Boom)
    composed_rec = _Recorder()
    monkeypatch.setattr(gdn_pkg, "ChunkGatedDeltaRuleFunction", composed_rec)
    with pytest.raises(RuntimeError, match="fused kernel missing"):
        gdn_pkg.ascendc_chunk_gated_delta_rule(
            q=torch.randn(1, 64, 2, 128),
            k=torch.randn(1, 64, 2, 128),
            v=torch.randn(1, 64, 2, 128),
            g=-torch.rand(1, 64, 2),
            beta=torch.rand(1, 64, 2),
            use_qk_l2norm_in_kernel=False,
            chunk_size=64,
        )
    assert len(composed_rec.calls) == 0


def test_save_backward_intermediates_is_rejected(monkeypatch):
    """The flag only ever served the composed chain; requesting it from a
    wrapper that never dispatches there fails fast."""
    q = torch.randn(1, 64, 2, 128)
    with pytest.raises(ValueError, match="save_backward_intermediates"):
        gdn_pkg.ascendc_chunk_gated_delta_rule(
            q=q,
            k=torch.randn_like(q),
            v=torch.randn_like(q),
            g=-torch.rand(1, 64, 2),
            beta=torch.rand(1, 64, 2),
            use_qk_l2norm_in_kernel=False,
            chunk_size=64,
            save_backward_intermediates=True,
        )


def test_no_dispatch_symbols_exist():
    """The probe/env machinery that would drive a fallback must not be
    reintroduced: re-adding it would silently reintroduce a composed-chain
    path."""
    assert not hasattr(gdn_pkg, "fused_gdn_available")
