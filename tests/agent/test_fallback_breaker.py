"""Tests for the paid-GLM fallback circuit breaker (t_eecca925).

Acceptance criteria:
  5. codex-up (default disabled breaker) path is unchanged.
  6. codex-down: first N metered calls allowed; N+1 skips to free fallback.
  7. Free (:free) fallback is never counted against the cap.
  8. Alert fires exactly once per outage window (not per over-cap call).
  9. Unreadable/corrupt state file → fail-open (glm proceeds, no exception).
"""
from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, call, patch

import pytest

from agent.fallback_breaker import (
    breaker_check_and_record,
    breaker_gate,
    is_metered,
    _fire_alert,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHAIN_GLM_THEN_LLAMA = [
    {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "z-ai/glm-5.2",
    },
    {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
    },
]

GLM_ENTRY = CHAIN_GLM_THEN_LLAMA[0]
LLAMA_ENTRY = CHAIN_GLM_THEN_LLAMA[1]

SMALL_CAP_CFG = {
    "enabled": True,
    "paid_call_cap": 3,
    "window_seconds": 3600,
    "state_file": "",  # overridden in fixtures
    "alert_target": "",
}


def _make_agent(fallback_model=None):
    """Minimal AIAgent with optional fallback chain."""
    from run_agent import AIAgent
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openai-codex.example/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


# ---------------------------------------------------------------------------
# is_metered
# ---------------------------------------------------------------------------

class TestIsMetered:
    def test_glm_is_metered(self):
        assert is_metered({"model": "z-ai/glm-5.2"}) is True

    def test_llama_free_is_not_metered(self):
        assert is_metered({"model": "meta-llama/llama-3.3-70b-instruct:free"}) is False

    def test_any_free_suffix(self):
        assert is_metered({"model": "some-model:free"}) is False

    def test_missing_model(self):
        assert is_metered({}) is True  # unknown → conservative (metered)


# ---------------------------------------------------------------------------
# Test 5: codex-up (disabled breaker) — path byte-identical
# ---------------------------------------------------------------------------

class TestBreakerDisabled:
    """When breaker is disabled (default), try_activate_fallback is unchanged."""

    def test_disabled_cfg_never_blocks(self, tmp_path):
        """breaker_check_and_record with enabled=False always returns (True, False)."""
        cfg = {"enabled": False, "paid_call_cap": 1, "window_seconds": 3600}
        sp = str(tmp_path / "breaker.json")
        allowed, just_opened = breaker_check_and_record(cfg, GLM_ENTRY, time.time(), sp)
        assert allowed is True
        assert just_opened is False

    def test_breaker_gate_disabled_by_default(self):
        """breaker_gate returns (True, False) when config has no enabled=true."""
        with patch("agent.fallback_breaker._load_cfg", return_value={"enabled": False}):
            allowed, just_opened = breaker_gate(None, GLM_ENTRY)
        assert allowed is True
        assert just_opened is False

    def test_agent_fallback_selects_glm_when_disabled(self):
        """Full integration: disabled breaker → glm entry still selected."""
        agent = _make_agent(fallback_model=CHAIN_GLM_THEN_LLAMA)

        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            return _mock_client(), model

        with (
            patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve),
            patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m),
            patch("agent.fallback_breaker._load_cfg", return_value={"enabled": False}),
        ):
            ok = agent._try_activate_fallback()

        assert ok is True
        assert agent.model == "z-ai/glm-5.2"


# ---------------------------------------------------------------------------
# Test 6: codex-down — cap, then free fallback
# ---------------------------------------------------------------------------

class TestBreakerCapThenFree:
    """First N metered calls are allowed; N+1 skips to free fallback."""

    def test_cap_then_skip_to_free(self, tmp_path):
        """Direct breaker_check_and_record: 3 allowed, 4th blocked."""
        cfg = {**SMALL_CAP_CFG, "state_file": str(tmp_path / "b.json")}
        sp = cfg["state_file"]
        t = time.time()

        for i in range(3):
            allowed, just_opened = breaker_check_and_record(cfg, GLM_ENTRY, t + i, sp)
            assert allowed is True, f"call {i + 1} should be allowed"

        allowed, just_opened = breaker_check_and_record(cfg, GLM_ENTRY, t + 3, sp)
        assert allowed is False
        assert just_opened is True  # first time cap hit → alert should fire

    def test_agent_skip_glm_pick_llama_after_cap(self, tmp_path):
        """Integration: 4th activation skips glm and lands on free llama."""
        agent = _make_agent(fallback_model=CHAIN_GLM_THEN_LLAMA)
        sp = str(tmp_path / "breaker.json")
        cfg = {**SMALL_CAP_CFG, "state_file": sp}

        # Save primary state to restore between activations.
        primary_model = agent.model
        primary_provider = getattr(agent, "provider", "")
        primary_base_url = str(getattr(agent, "base_url", ""))

        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            return _mock_client(), model

        selected_models = []

        def activate():
            # Reset to primary state so dedup logic doesn't skip glm on re-entry.
            agent.model = primary_model
            agent.provider = primary_provider
            agent.base_url = primary_base_url
            agent._fallback_index = 0
            agent._fallback_activated = False
            with (
                patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve),
                patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m),
                patch("agent.fallback_breaker._load_cfg", return_value=cfg),
                patch("agent.fallback_breaker._fire_alert"),
            ):
                ok = agent._try_activate_fallback()
            selected_models.append(agent.model)
            return ok

        # Calls 1–3: GLM should be selected (within cap).
        for i in range(3):
            assert activate() is True, f"call {i + 1} should succeed"
            assert selected_models[-1] == "z-ai/glm-5.2", f"call {i + 1}: expected glm, got {selected_models[-1]}"

        # Call 4: GLM is capped → should select free llama.
        assert activate() is True
        assert selected_models[-1] == "meta-llama/llama-3.3-70b-instruct:free", (
            f"4th call should pick free llama, got {selected_models[-1]}"
        )


# ---------------------------------------------------------------------------
# Test 7: free fallback never counted
# ---------------------------------------------------------------------------

class TestFreeNotCounted:
    def test_free_never_decrements_cap(self, tmp_path):
        """Activating :free entries must not increment the metered call counter."""
        cfg = {**SMALL_CAP_CFG, "state_file": str(tmp_path / "b.json"), "paid_call_cap": 1}
        sp = cfg["state_file"]
        t = time.time()

        # 10 free activations — should never trip the cap.
        for i in range(10):
            allowed, just_opened = breaker_check_and_record(cfg, LLAMA_ENTRY, t + i, sp)
            assert allowed is True
            assert just_opened is False

        # State file should still be empty or have zero calls.
        if os.path.exists(sp):
            with open(sp) as f:
                state = json.load(f)
            assert len(state.get("calls", [])) == 0

    def test_breaker_gate_free_fast_path(self):
        with patch("agent.fallback_breaker._load_cfg", return_value={**SMALL_CAP_CFG}):
            allowed, just_opened = breaker_gate(None, LLAMA_ENTRY)
        assert allowed is True
        assert just_opened is False


# ---------------------------------------------------------------------------
# Test 8: alert fires exactly once per window
# ---------------------------------------------------------------------------

class TestAlertOnce:
    def test_alert_once_per_window(self, tmp_path):
        """_fire_alert invoked exactly once across many over-cap activations."""
        cfg = {**SMALL_CAP_CFG, "state_file": str(tmp_path / "b.json")}
        sp = cfg["state_file"]
        t = time.time()

        # Fill up to cap.
        for i in range(3):
            breaker_check_and_record(cfg, GLM_ENTRY, t + i, sp)

        fire_calls = []
        with patch("agent.fallback_breaker._fire_alert", side_effect=lambda *a, **k: fire_calls.append(1)):
            # 5 over-cap calls via breaker_gate — alert should fire exactly once.
            for i in range(5):
                with patch("agent.fallback_breaker._load_cfg", return_value=cfg):
                    breaker_gate(None, GLM_ENTRY)

        assert len(fire_calls) == 1, (
            f"Expected exactly 1 alert across 5 over-cap calls, got {len(fire_calls)}"
        )

    def test_alert_resets_after_window_expires(self, tmp_path):
        """After window expires (all calls age out), next cap hit fires alert again."""
        cfg = {**SMALL_CAP_CFG, "state_file": str(tmp_path / "b.json"),
               "window_seconds": 60}
        sp = cfg["state_file"]
        t0 = 1000.0

        # Fill cap and trip breaker (first alert).
        for i in range(3):
            breaker_check_and_record(cfg, GLM_ENTRY, t0 + i, sp)
        _, just_opened_1 = breaker_check_and_record(cfg, GLM_ENTRY, t0 + 3, sp)
        assert just_opened_1 is True

        # Advance time past the window — all prior calls expire.
        t1 = t0 + 120  # 2× window

        # Fill cap again in the new window.
        for i in range(3):
            allowed, _ = breaker_check_and_record(cfg, GLM_ENTRY, t1 + i, sp)
            assert allowed is True

        # 4th call in new window should open breaker again (just_opened=True).
        _, just_opened_2 = breaker_check_and_record(cfg, GLM_ENTRY, t1 + 3, sp)
        assert just_opened_2 is True, "Alert should fire again after window reset"


# ---------------------------------------------------------------------------
# Test 9: fail-open on corrupt / unreadable state
# ---------------------------------------------------------------------------

class TestFailOpen:
    def test_corrupt_state_file_fail_open(self, tmp_path):
        """Corrupt JSON → breaker fails open (allowed=True, no exception)."""
        sp = str(tmp_path / "b.json")
        with open(sp, "w") as f:
            f.write("not valid json }{")

        cfg = {**SMALL_CAP_CFG, "state_file": sp}
        allowed, just_opened = breaker_check_and_record(cfg, GLM_ENTRY, time.time(), sp)
        assert allowed is True
        assert just_opened is False

    def test_unreadable_state_dir_fail_open(self, tmp_path):
        """Non-existent dir auto-created; if dir creation fails → fail-open."""
        # Point to a path whose parent dir is a file (creation fails).
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file, not a dir")
        sp = str(blocker / "state.json")  # parent is a file → makedirs fails

        cfg = {**SMALL_CAP_CFG, "state_file": sp}
        allowed, just_opened = breaker_check_and_record(cfg, GLM_ENTRY, time.time(), sp)
        assert allowed is True

    def test_breaker_gate_import_error_fail_open(self):
        """If breaker_gate itself raises (simulated), fallback still proceeds."""
        with patch("agent.fallback_breaker._load_cfg", side_effect=RuntimeError("injected")):
            allowed, just_opened = breaker_gate(None, GLM_ENTRY)
        assert allowed is True
        assert just_opened is False
