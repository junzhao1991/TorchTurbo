"""Fused GDN dispatch tests: the probe drives the wrapper decision.

These tests mock the ``fused_gdn_usable`` probe (a thin reader of one
lru_cached `_probe_fused_pair` result) to cover the dispatch cases without
running any real fused kernel:

* probe true                -> FusedFunction (fwd+bwd both fused)
* probe false               -> composed ChunkGatedDeltaRuleFunction, both
                              directions (no mixed granularity; the fused
                              backward needs the fused forward's exports)
* probe true but outside the fused envelope (V=256) -> composed chain
* ``_USE_FUSED_GDN`` escape hatch off -> composed chain

The Function classes themselves are replaced by recorders, so nothing
launches on device; only the wrapper's dispatch logic (argument passing)
is under test. A separate test locks the probe's once-per-process caching
semantics against ``_probe_pair``-style failures.

These tests import the vendored adapter (which needs ``fla_npu`` /
``torch_npu``), so they run on an NPU install; the importorskip guards make
them a no-op elsewhere.
"""

import pytest
from functools import lru_cache

pytest.importorskip("torch_npu")
pytest.importorskip("fla_npu")

import torch  # noqa: E402

import torchturbo.ops.providers.npu._fla_gdn as gdn_pkg  # noqa: E402
import torchturbo.ops.providers.npu._fla_gdn.chunk_gated_delta_rule_function as gdn_fn  # noqa: E402


def _fake_tensors(batch=1, tokens=1057, value_dim=128):
    """Time-first inputs in the wrapper's fla-like calling convention."""
    q = torch.randn(batch, tokens, 16, 128)
    k = torch.randn(batch, tokens, 16, 128)
    v = torch.randn(batch, tokens, 32, value_dim)
    g = -torch.rand(batch, tokens, 32)
    beta = torch.rand(batch, tokens, 32)
    return q, k, v, g, beta


class _Recorder:
    """Mimics a torch.autograd.Function: ``.apply(...)`` records the call."""

    def __init__(self, result=(torch.zeros(1), None)):
        self.result = result
        self.calls = []

    def apply(self, *args, **kwargs):  # noqa: D102 (stands in for Function)
        self.calls.append((args, kwargs))
        return self.result


@pytest.fixture
def clean_probe_cache():
    """Clear the probe cache around every test.

    ``fused_gdn_usable`` reads one lru_cached `_probe_fused_pair` result.
    Tests monkeypatching the probe on ``gdn_pkg`` bypass it entirely (the
    wrapper calls the patched callable), but tests monkeypatching on
    ``gdn_fn`` still hit the cache through the real ``fused_gdn_usable``,
    so the reset keeps them order-independent.
    """

    def _clear():
        cached = getattr(gdn_fn._probe_fused_pair, "cache_clear", None)
        if cached is not None:  # monkeypatched probes have no cache
            cached()

    _clear()
    yield
    _clear()


def _run_wrapper(monkeypatch, tensors, use_fused=True, **extra):
    fused_rec = _Recorder()
    composed_rec = _Recorder()
    monkeypatch.setattr(gdn_pkg, "ChunkGatedDeltaRuleFusedFunction", fused_rec)
    monkeypatch.setattr(gdn_pkg, "ChunkGatedDeltaRuleFunction", composed_rec)
    if use_fused is not None:
        monkeypatch.setattr(gdn_pkg, "_USE_FUSED_GDN", use_fused)
    q, k, v, g, beta = tensors
    o, final_state = gdn_pkg.ascendc_chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        use_qk_l2norm_in_kernel=False,
        chunk_size=64,
        **extra,
    )
    return fused_rec, composed_rec, o, final_state


def test_probe_true_full_fusion(monkeypatch, clean_probe_cache):
    """Both directions fused: one FusedFunction call, no composed call."""
    monkeypatch.setattr(gdn_pkg, "fused_gdn_usable", lambda: True)
    fused, composed, *_ = _run_wrapper(monkeypatch, _fake_tensors())
    assert len(fused.calls) == 1
    assert len(composed.calls) == 0
    args = fused.calls[0][0]
    assert len(args) == 14  # composed-chain 14 inputs (no save/flag extras)
    # q/k/v were transposed to head-first [B,H,T,D] on the way in
    assert tuple(args[0].shape) == (1, 16, 1057, 128)
    assert tuple(args[2].shape) == (1, 32, 1057, 128)
    assert args[5] == pytest.approx(128 ** -0.5)  # scale defaults to dk ** -0.5
    assert args[13] == 64  # chunk_size


def test_probe_failed_full_fallback(monkeypatch, clean_probe_cache):
    """A failed probe falls to the composed chain for both directions --
    no mixed granularity (fused forward with composed backward)."""
    monkeypatch.setattr(gdn_pkg, "fused_gdn_usable", lambda: False)
    fused, composed, *_ = _run_wrapper(monkeypatch, _fake_tensors())
    assert len(fused.calls) == 0
    assert len(composed.calls) == 1
    args = composed.calls[0][0]
    assert len(args) == 15  # original 14-input signature + save_intermediates
    assert isinstance(args[-1], bool)  # save_backward_intermediates slot


def test_outside_envelope_ignores_probe(monkeypatch, clean_probe_cache):
    """V=256 is in the fused fwd's own envelope but outside the fused
    pair's (K=V=128), so the wrapper must stay composed even with the
    probe green."""
    monkeypatch.setattr(gdn_pkg, "fused_gdn_usable", lambda: True)
    fused, composed, *_ = _run_wrapper(
        monkeypatch, _fake_tensors(value_dim=256)
    )
    assert len(fused.calls) == 0
    assert len(composed.calls) == 1


def test_escape_hatch_disables_fusion(monkeypatch, clean_probe_cache):
    """``_USE_FUSED_GDN`` (env TORCHTURBO_GDN_USE_FUSED=0) forced off keeps
    the wrapper on the composed chain regardless of the probe."""
    monkeypatch.setattr(gdn_pkg, "fused_gdn_usable", lambda: True)
    fused, composed, *_ = _run_wrapper(
        monkeypatch, _fake_tensors(), use_fused=False
    )
    assert len(fused.calls) == 0
    assert len(composed.calls) == 1


def test_probe_cached_once_per_process(monkeypatch, clean_probe_cache):
    """The fused-pair probe runs its tiny launch at most once, however
    often it is asked (lru_cache around the probe)."""
    probe_calls = []

    @lru_cache(maxsize=1)
    def fake_probe():
        probe_calls.append(True)
        return True

    monkeypatch.setattr(gdn_fn, "_probe_fused_pair", fake_probe)
    # fused_gdn_usable reads the module-global _probe_fused_pair at call
    # time, so the patch above is what it executes.
    assert gdn_fn.fused_gdn_usable() is True
    assert gdn_fn.fused_gdn_usable() is True
    assert gdn_fn.fused_gdn_usable() is True
    assert probe_calls == [True]


def test_probe_failure_caches_unusable(monkeypatch, clean_probe_cache):
    """A launch failure inside the probe returns
    False; the cached result must mark the pair unusable, never crash the
    caller -- importability alone must not mark it usable."""
    monkeypatch.setattr(gdn_fn, "_probe_fused_pair", lambda: False)
    assert gdn_fn.fused_gdn_usable() is False
    assert gdn_fn.fused_gdn_usable() is False


def test_fused_envelope_supported():
    assert gdn_fn.fused_envelope_supported(128, 128, 64) is True
    assert gdn_fn.fused_envelope_supported(128, 256, 64) is False
    assert gdn_fn.fused_envelope_supported(128, 128, 128) is False
    assert gdn_fn.fused_envelope_supported(64, 128, 64) is False
