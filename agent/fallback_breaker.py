"""fallback_breaker.py — circuit breaker for paid fallback providers during outages.

Prevents unbounded spend on metered (non-:free) fallback providers during a
primary-model outage. When the paid-GLM call count exceeds the configured cap
within a rolling window, the breaker opens: subsequent calls skip the metered
entry and recurse to the next (free) fallback in the chain.

Background: 2026-06-29 incident — codex stuck client in gateway caused ~1h45m outage,
draining ~$16 into 513 paid z-ai/glm-5.2 fallback calls. This circuit breaker is the
backstop; the self-heal restart in codex-health-watch.py is the primary fix.

State is file-based so the cap applies across all agent processes (cron + gateway).
Uses best-effort flock for cross-process safety — imperfect locking may allow a few
extra calls past the cap, but still bounds the bleed by ~10x.

Fail-open: any I/O, lock, or import error returns (allowed=True, just_opened=False)
so the fallback chain is never blocked by breaker failure.

Configuration (~/.hermes/config.yaml):
    fallback_circuit_breaker:
      enabled: true
      paid_call_cap: 50          # max metered activations per window (default: 50)
      window_seconds: 3600       # rolling window length in seconds (default: 3600)
      state_file: ""             # "" → ~/.hermes/state/fallback-breaker.json
      alert_target: ""           # "" → telegram:552296

Paid provider detection: any model WITHOUT ':free' suffix is considered metered.
This correctly classifies the current chain:
  z-ai/glm-5.2                          → metered (no :free)
  meta-llama/llama-3.3-70b-instruct:free → free (has :free)
If a new paid fallback is added later it is automatically covered by this heuristic.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_ALERT_TARGET = "telegram:552296"
_DEFAULT_STATE_DIR = os.path.join(os.path.expanduser("~"), ".hermes", "state")
_DEFAULT_STATE_FILE = os.path.join(_DEFAULT_STATE_DIR, "fallback-breaker.json")

# Simple config cache keyed on config-file mtime to avoid re-reading on every call.
_CFG_CACHE: Optional[Dict[str, Any]] = None
_CFG_MTIME: float = -1.0
# One-shot guard so a persistent config-load failure (which silently disables
# the breaker → money can bleed unbounded) is logged once, not on every call.
_CFG_LOAD_WARNED: bool = False


def is_metered(entry: Dict[str, Any]) -> bool:
    """Return True if the fallback entry uses a metered (non-free) model.

    Convention from OpenRouter: free models carry a ':free' suffix.
    Examples: z-ai/glm-5.2 → metered; meta-llama/llama-3.3-70b-instruct:free → free.
    """
    model = (entry.get("model") or "").lower()
    return ":free" not in model


def _load_cfg() -> Dict[str, Any]:
    """Load breaker section from hermes config. Cached by config-file mtime."""
    global _CFG_CACHE, _CFG_MTIME, _CFG_LOAD_WARNED
    try:
        from hermes_cli.config import load_config_readonly, get_config_path
        cfg_path = get_config_path()
        try:
            mtime = cfg_path.stat().st_mtime
        except OSError:
            mtime = -1.0
        if _CFG_CACHE is None or mtime != _CFG_MTIME:
            cfg = load_config_readonly()
            _CFG_CACHE = dict(cfg.get("fallback_circuit_breaker") or {})
            _CFG_MTIME = mtime
        _CFG_LOAD_WARNED = False  # successful load clears the one-shot warning
        return _CFG_CACHE
    except Exception as e:
        # Fail-open returns {} → breaker disabled → paid fallback uncapped.
        # That is a money-relevant degradation, so make it visible at least once
        # instead of failing silently (was a silent path; review M3).
        if not _CFG_LOAD_WARNED:
            logger.warning(
                "fallback_breaker: config load failed — breaker DISABLED, paid "
                "fallback is uncapped until config loads again: %s", e,
            )
            _CFG_LOAD_WARNED = True
        return {}


def _state_path(cfg: Dict[str, Any]) -> str:
    p = (cfg.get("state_file") or "").strip()
    return os.path.expanduser(p) if p else _DEFAULT_STATE_FILE


def _ensure_state_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _alert_target(cfg: Dict[str, Any]) -> str:
    t = (cfg.get("alert_target") or "").strip()
    return t or _DEFAULT_ALERT_TARGET


def _fire_alert(cfg: Dict[str, Any], cap: int, window: int) -> None:
    """Send one-time Telegram alert that the paid-GLM circuit breaker has opened."""
    target = _alert_target(cfg)
    hermes = os.path.join(os.path.expanduser("~"), ".local/bin/hermes")
    subject = "⚡ Paid-GLM лимит достигнут — circuit breaker открыт"
    body = (
        f"За последние {window // 60} мин было >{cap} платных GLM-фолбэков во время "
        "недоступности codex. Переключаюсь на бесплатный llama-fallback до восстановления codex. "
        "Это уведомление — один раз за окно, спама не будет."
    )
    if os.environ.get("HERMES_BREAKER_DRY_SEND", "").strip() == "1":
        print(f"BREAKER_ALERT[{target}] {subject}: {body}")
        return
    try:
        subprocess.run(
            [hermes, "send", "-t", target, "-s", subject, body],
            check=False, timeout=30, capture_output=True,
        )
    except Exception as e:
        logger.warning("fallback_breaker: alert send failed: %s", e)


def breaker_check_and_record(
    cfg: Dict[str, Any],
    entry: Dict[str, Any],
    now: float,
    state_path: str,
) -> Tuple[bool, bool]:
    """Check whether this metered-fallback call is allowed; record it if so.

    Returns (allowed, just_opened):
      allowed=True  → proceed with this fallback entry
      just_opened=True → cap just hit for the first time this window; caller fires alert

    Always fail-open: any I/O or lock error returns (True, False).
    """
    if not cfg.get("enabled", False):
        return True, False
    if not is_metered(entry):
        return True, False

    cap = int(cfg.get("paid_call_cap", 50))
    window = int(cfg.get("window_seconds", 3600))

    fp = None
    try:
        _ensure_state_dir(state_path)
        fp = open(state_path, "a+")  # create if missing, open for read+append
        try:
            fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another process holds the lock; fail-open rather than block the agent.
            logger.debug("fallback_breaker: state lock contended, failing open")
            return True, False

        fp.seek(0)
        raw = fp.read().strip()
        try:
            state: Dict[str, Any] = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, ValueError):
            state = {}

        # Prune calls older than the rolling window.
        calls = [t for t in (state.get("calls") or []) if now - t < window]
        alerted = bool(state.get("alerted", False))

        # Reset alerted when all prior calls have expired (new outage window).
        if not calls:
            alerted = False

        if len(calls) >= cap:
            just_opened = not alerted
            state["alerted"] = True
            state["calls"] = calls
            _write_breaker_state(fp, state)
            return False, just_opened

        calls.append(now)
        state["calls"] = calls
        state["alerted"] = alerted
        _write_breaker_state(fp, state)
        return True, False
    except Exception as e:
        logger.warning("fallback_breaker: state error (fail-open): %s", e)
        return True, False
    finally:
        if fp is not None:
            try:
                fp.close()
            except Exception:
                pass


def _write_breaker_state(fp: Any, state: Dict[str, Any]) -> None:
    """Overwrite the locked state file with new content.

    Serialize to a string first, then write+flush+fsync in one shot so a
    crash mid-write is far less likely to leave torn JSON.  A torn file would
    decode-error → reset the counter → silently re-arm the cap (money-relevant),
    so durability here directly bounds spend.  The flock is held on ``fp`` by
    the caller, so an in-place rewrite is safe for cross-process correctness.
    """
    data = json.dumps(state)
    fp.seek(0)
    fp.truncate()
    fp.write(data)
    fp.flush()
    try:
        os.fsync(fp.fileno())
    except OSError:
        pass


def breaker_gate(agent: Any, fb: Dict[str, Any]) -> Tuple[bool, bool]:
    """Entry point called from try_activate_fallback after the fallback entry is chosen.

    Returns (allowed, just_opened). Always fail-open on any error.
    When just_opened=True the caller should fire the once-per-window alert.
    """
    try:
        cfg = _load_cfg()
        if not cfg.get("enabled", False):
            return True, False
        if not is_metered(fb):
            return True, False

        path = _state_path(cfg)
        allowed, just_opened = breaker_check_and_record(cfg, fb, time.time(), path)
        if just_opened:
            cap = int(cfg.get("paid_call_cap", 50))
            window = int(cfg.get("window_seconds", 3600))
            _fire_alert(cfg, cap, window)
        return allowed, just_opened
    except Exception as e:
        logger.warning("fallback_breaker: breaker_gate error (fail-open): %s", e)
        return True, False
