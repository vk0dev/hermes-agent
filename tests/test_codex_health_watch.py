"""Tests for codex-health-watch.py self-heal behavior (t_eecca925, PART 1).

Uses env-override simulation mode so NO real Telegram sends or launchctl restarts
are issued. Each test drives the script as a subprocess with temp log/state files.

Acceptance criteria:
  1. DOWN→one restart: seeded log has codex FAIL newer than OK; fresh state.
     → kickstart cmd printed, DM printed, state.restart_count=1, state.state=down.
  2. Cooldown blocks 2nd: same down log, now=t0+10min.
     → no kickstart, no DM, restart_count unchanged.
  3. Cooldown elapsed → restart again: now=t0+31min.
     → kickstart printed, restart_count=2.
  4. Recovery: log has codex OK newest; run.
     → recovery DM (no billing wording), state reset.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = str(Path("~/.hermes/scripts/codex-health-watch.py").expanduser())


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _make_log(lines: list[str], tmp_path: Path) -> Path:
    p = tmp_path / "agent.log"
    p.write_text("\n".join(lines) + "\n")
    return p


def _make_state(data: dict, tmp_path: Path) -> Path:
    p = tmp_path / "state.json"
    p.write_text(json.dumps(data))
    return p


def _run(log_path: str, state_path: str, now_iso: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update({
        "CODEX_HEALTH_LOG": log_path,
        "CODEX_HEALTH_STATE": state_path,
        "CODEX_HEALTH_NOW": now_iso,
        "CODEX_HEALTH_DRY_RESTART": "1",
        "CODEX_HEALTH_DRY_SEND": "1",
        "CODEX_HEALTH_UID": "501",
    })
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True, text=True, env=env,
    )


def _read_state(state_path: str) -> dict:
    with open(state_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Test 1: UP→DOWN — first auto restart
# ---------------------------------------------------------------------------

class TestDownFirstRestart:
    def test_restart_and_dm_on_first_down(self, tmp_path):
        t0 = datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc)
        log = _make_log([
            f"{_ts(t0 - timedelta(minutes=30))} INFO API call #1: model=gpt-5.5 provider=openai-codex some=stuff",
            f"{_ts(t0)} ERROR API call failed attempt=1 provider=openai-codex error=connection",
        ], tmp_path)
        state_file = str(tmp_path / "state.json")
        # No pre-existing state → fresh
        result = _run(str(log), state_file, t0.isoformat())
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output = result.stdout

        # Kickstart command should appear
        assert "DRY_RESTART: launchctl kickstart -k gui/501/ai.hermes.gateway" in output
        # DM should appear
        assert "DM[" in output
        assert "авто-рестарт" in output.lower() or "CODEX" in output

        st = _read_state(state_file)
        assert st["state"] == "down"
        assert st["restart_count"] == 1
        assert st["last_restart_at"] is not None
        assert st["down_since"] is not None

    def test_no_billing_assumption_in_dm(self, tmp_path):
        t0 = datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc)
        log = _make_log([
            f"{_ts(t0)} ERROR API call failed attempt=1 provider=openai-codex error=connection",
        ], tmp_path)
        state_file = str(tmp_path / "state.json")
        result = _run(str(log), state_file, t0.isoformat())
        output = result.stdout
        # Must NOT say anything about subscriptions/billing/payment
        assert "подписк" not in output.lower()
        assert "billing" not in output.lower()
        assert "ChatGPT" not in output


# ---------------------------------------------------------------------------
# Test 2: Cooldown blocks second restart
# ---------------------------------------------------------------------------

class TestCooldownBlocks:
    def test_silent_during_cooldown(self, tmp_path):
        t0 = datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc)
        log = _make_log([
            f"{_ts(t0)} ERROR API call failed attempt=1 provider=openai-codex error=conn",
        ], tmp_path)
        # Pre-existing state: already restarted at t0
        state_file = _make_state({
            "state": "down",
            "down_since": t0.isoformat(),
            "alert_count": 1,
            "last_restart_at": t0.isoformat(),
            "restart_count": 1,
        }, tmp_path)

        t1 = t0 + timedelta(minutes=10)  # well within 30-min cooldown
        result = _run(str(log), str(state_file), t1.isoformat())
        assert result.returncode == 0
        output = result.stdout

        assert "DRY_RESTART" not in output
        assert "DM[" not in output

        st = _read_state(str(state_file))
        assert st["restart_count"] == 1  # unchanged


# ---------------------------------------------------------------------------
# Test 3: Cooldown elapsed — second restart
# ---------------------------------------------------------------------------

class TestCooldownExpired:
    def test_restart_after_cooldown(self, tmp_path):
        t0 = datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc)
        log = _make_log([
            f"{_ts(t0)} ERROR API call failed attempt=1 provider=openai-codex error=conn",
        ], tmp_path)
        state_file = _make_state({
            "state": "down",
            "down_since": t0.isoformat(),
            "alert_count": 1,
            "last_restart_at": t0.isoformat(),
            "restart_count": 1,
        }, tmp_path)

        t1 = t0 + timedelta(minutes=31)  # past the 30-min cooldown
        result = _run(str(log), str(state_file), t1.isoformat())
        assert result.returncode == 0
        output = result.stdout

        assert "DRY_RESTART: launchctl kickstart -k gui/501/ai.hermes.gateway" in output
        assert "DM[" in output

        st = _read_state(str(state_file))
        assert st["restart_count"] == 2


# ---------------------------------------------------------------------------
# Test 4: Recovery DOWN→UP
# ---------------------------------------------------------------------------

class TestRecovery:
    def test_recovery_note_on_up(self, tmp_path):
        t0 = datetime(2026, 6, 29, 10, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc)
        log = _make_log([
            f"{_ts(t0)} ERROR API call failed attempt=1 provider=openai-codex error=conn",
            f"{_ts(t1)} INFO API call #5: model=gpt-5.5 provider=openai-codex tokens=100",
        ], tmp_path)
        state_file = _make_state({
            "state": "down",
            "down_since": t0.isoformat(),
            "alert_count": 1,
            "last_restart_at": t0.isoformat(),
            "restart_count": 1,
        }, tmp_path)

        result = _run(str(log), str(state_file), t1.isoformat())
        assert result.returncode == 0
        output = result.stdout

        # Recovery DM should be sent
        assert "DM[" in output
        assert "восстановлен" in output.lower() or "CODEX" in output

        # No billing wording in recovery either
        assert "подписк" not in output.lower()
        assert "ChatGPT" not in output

        # No restart attempted on recovery
        assert "DRY_RESTART" not in output

        st = _read_state(str(state_file))
        assert st["state"] == "up"
        assert st["last_restart_at"] is None
        assert st["restart_count"] == 0

    def test_state_fully_reset_on_recovery(self, tmp_path):
        t0 = datetime(2026, 6, 29, 10, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc)
        log = _make_log([
            # Codex OK is NEWER than the fail — system is up
            f"{_ts(t0)} ERROR API call failed attempt=1 provider=openai-codex error=conn",
            f"{_ts(t1)} INFO API call #5: model=gpt-5.5 provider=openai-codex tokens=100",
        ], tmp_path)
        state_file = _make_state({"state": "down", "down_since": t0.isoformat(),
                                   "alert_count": 3, "last_restart_at": t0.isoformat(),
                                   "restart_count": 2}, tmp_path)
        _run(str(log), str(state_file), t1.isoformat())

        st = _read_state(str(state_file))
        assert st == {
            "state": "up",
            "down_since": None,
            "alert_count": 0,
            "last_restart_at": None,
            "restart_count": 0,
        }
