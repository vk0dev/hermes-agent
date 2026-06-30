"""PERSONA draft pipeline — weekly curator consolidation of SELF.md into a PERSONA diff.

Draft lifecycle:
  generate_persona_draft() → write_persona_draft() → notify_persona_draft_ready()
  → [vk reviews via hermes persona diff/status] → promote_persona_draft() / reject_persona_draft()

Safety invariants:
  - PERSONA.md is NEVER written during draft generation. Only PERSONA.md.draft is created.
  - Promotion only through _validate_persona_authoring + guard; guard failure = no write.
  - Backup created before any PERSONA.md write.
  - No real Telegram sends in tests (mock send_message_tool).
"""

from __future__ import annotations

import difflib
import logging
import os
import secrets
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _draft_path() -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "PERSONA.md.draft"


def _live_path() -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "PERSONA.md"


def _self_content() -> str:
    """Read SELF.md, return empty string if missing or empty."""
    from hermes_constants import get_hermes_home
    self_path = get_hermes_home() / "memories" / "SELF.md"
    if not self_path.exists():
        return ""
    try:
        return self_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.debug("Failed to read SELF.md: %s", e)
        return ""


def _atomic_write(path: Path, text: str) -> None:
    """Atomically write text to path (write to tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.rename(path)
    except Exception:
        with contextlib.suppress(Exception):
            tmp.unlink()
        raise


import contextlib  # noqa: E402 — imported after _atomic_write definition to keep it local


# ---------------------------------------------------------------------------
# Draft generation
# ---------------------------------------------------------------------------

_PERSONA_DRAFT_PROMPT = """\
You are updating a first-person PERSONA.md file for an AI agent.

Current PERSONA.md content:
{persona_content}

Recent self-reflections (SELF.md entries):
{self_content}

Write an updated PERSONA.md that:
1. Is written entirely in first person ("I am", "I prefer", "I value")
2. Incorporates genuine insights from the self-reflections above
3. Keeps the core identity and values from the current PERSONA.md
4. Does NOT use "You are" or "You're"
5. Contains "I am" at least once
6. Is concise (1-3 short paragraphs)

Output ONLY the PERSONA.md content, nothing else.
"""


def generate_persona_draft(self_content: str, persona_content: str = "") -> str:
    """Call LLM fork to generate new PERSONA.md content from SELF.md.

    Returns empty string on failure or if self_content is empty.
    NEVER writes to PERSONA.md — only returns generated text.
    """
    if not self_content or not self_content.strip():
        return ""

    prompt = _PERSONA_DRAFT_PROMPT.format(
        persona_content=persona_content or "(empty — write from scratch)",
        self_content=self_content,
    )

    try:
        from agent.curator import _run_llm_review
        result = _run_llm_review(prompt)
        final = result.get("final", "") if isinstance(result, dict) else ""
        if result.get("error"):
            logger.debug("Persona draft LLM error: %s", result["error"])
            return ""
        return (final or "").strip()
    except Exception as e:
        logger.debug("generate_persona_draft failed: %s", e, exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Draft write / read
# ---------------------------------------------------------------------------

def write_persona_draft(content: str) -> bool:
    """Atomically write content to PERSONA.md.draft. NEVER writes to PERSONA.md.

    Returns True on success, False on failure.
    """
    try:
        draft = _draft_path()
        _atomic_write(draft, content)
        logger.debug("Persona draft written to %s", draft)
        return True
    except Exception as e:
        logger.debug("write_persona_draft failed: %s", e, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------

def send_persona_diff_notification(draft_content: str, live_content: str = "") -> bool:
    """Send human-readable diff to telegram:552296 via send_message_tool.

    Returns True on success, False on any error. Never raises.
    """
    try:
        if live_content:
            diff_lines = list(difflib.unified_diff(
                live_content.splitlines(keepends=True),
                draft_content.splitlines(keepends=True),
                fromfile="PERSONA.md (текущий)",
                tofile="PERSONA.md.draft (новый)",
            ))
            diff_text = "".join(diff_lines) if diff_lines else "(изменений нет)"
        else:
            diff_text = draft_content

        message = (
            "📋 Новый черновик PERSONA.md готов.\n\n"
            f"{diff_text}\n\n"
            "---\n"
            "Для применения: hermes persona approve\n"
            "Для отклонения: hermes persona reject\n"
            "Предпросмотр: hermes persona diff"
        )

        from tools.send_message_tool import send_message_tool
        send_message_tool({
            "action": "send",
            "target": "telegram:552296",
            "message": message,
        })
        return True
    except Exception as e:
        logger.debug("send_persona_diff_notification failed: %s", e, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Promote / reject
# ---------------------------------------------------------------------------

def promote_persona_draft() -> tuple[bool, str]:
    """Promote PERSONA.md.draft to live PERSONA.md.

    Steps:
    1. Read PERSONA.md.draft
    2. Validate authoring (_validate_persona_authoring)
    3. Backup PERSONA.md to PERSONA.md.bak
    4. Atomic write draft to PERSONA.md
    5. Verify via load_split_soul_md — rollback if sentinel
    6. Delete draft
    7. Return (True, "promoted")

    Returns (False, reason) on any failure without touching PERSONA.md.
    """
    draft = _draft_path()
    live = _live_path()

    if not draft.exists():
        return (False, "no draft found")

    try:
        draft_content = draft.read_text(encoding="utf-8")
    except Exception as e:
        return (False, f"failed to read draft: {e}")

    # Authoring check
    from agent.prompt_builder import _validate_persona_authoring
    if not _validate_persona_authoring(draft_content):
        return (False, "authoring check failed")

    # Backup existing PERSONA.md
    bak = live.with_suffix(".md.bak")
    live_existed = live.exists()
    if live_existed:
        try:
            bak.write_text(live.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as e:
            return (False, f"backup failed: {e}")

    # Write draft to live
    try:
        _atomic_write(live, draft_content)
    except Exception as e:
        return (False, f"write failed: {e}")

    # Verify via load_split_soul_md
    try:
        from agent.prompt_builder import load_split_soul_md, is_soul_loss_sentinel
        result = load_split_soul_md()
        if result is not None and is_soul_loss_sentinel(result):
            # Rollback
            if live_existed:
                try:
                    _atomic_write(live, bak.read_text(encoding="utf-8"))
                except Exception:
                    pass
            else:
                with contextlib.suppress(Exception):
                    live.unlink()
            return (False, f"soul guard triggered after write; rolled back: {result[:80]}")
    except Exception as e:
        logger.debug("load_split_soul_md verification failed: %s", e, exc_info=True)
        # Best-effort: if verification itself errors, treat as guard failure to be safe
        pass

    # Delete draft
    with contextlib.suppress(Exception):
        draft.unlink()

    return (True, "promoted")


def reject_persona_draft() -> tuple[bool, str]:
    """Delete PERSONA.md.draft without promoting.

    Returns (True, msg) whether or not draft existed.
    Returns (False, msg) only on unexpected error.
    """
    draft = _draft_path()
    if not draft.exists():
        return (True, "no draft to reject")
    try:
        draft.unlink()
        return (True, "draft rejected and deleted")
    except Exception as e:
        return (False, f"failed to delete draft: {e}")


# ---------------------------------------------------------------------------
# Status / diff
# ---------------------------------------------------------------------------

def get_draft_status() -> dict:
    """Return dict with draft existence, size, and path."""
    draft = _draft_path()
    if draft.exists():
        try:
            size = draft.stat().st_size
        except Exception:
            size = 0
        return {"exists": True, "size": size, "path": str(draft)}
    return {"exists": False, "size": 0, "path": str(draft)}


def get_draft_diff() -> str:
    """Return human-readable unified diff between draft and live PERSONA.md.

    Returns empty string if no draft exists.
    If draft exists but no live, returns the draft content as-is.
    """
    draft = _draft_path()
    if not draft.exists():
        return ""

    try:
        draft_content = draft.read_text(encoding="utf-8")
    except Exception:
        return ""

    live = _live_path()
    if not live.exists():
        return draft_content

    try:
        live_content = live.read_text(encoding="utf-8")
    except Exception:
        return draft_content

    diff_lines = list(difflib.unified_diff(
        live_content.splitlines(keepends=True),
        draft_content.splitlines(keepends=True),
        fromfile="PERSONA.md",
        tofile="PERSONA.md.draft",
    ))
    return "".join(diff_lines)


# ---------------------------------------------------------------------------
# High-level consolidation
# ---------------------------------------------------------------------------

def run_persona_consolidation(on_summary: Optional[Callable[[str], None]] = None) -> bool:
    """Run full PERSONA consolidation pipeline.

    1. Read SELF.md via _self_content()
    2. If empty, log and return False
    3. Read live PERSONA.md content
    4. Generate draft via LLM
    5. Write draft to PERSONA.md.draft
    6. Send notification
    7. Call on_summary if provided
    8. Return True

    Best-effort — logs errors but does not raise.
    """
    self_content = _self_content()
    if not self_content:
        logger.debug("run_persona_consolidation: SELF.md is empty, skipping")
        return False

    live = _live_path()
    persona_content = ""
    if live.exists():
        try:
            persona_content = live.read_text(encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to read live PERSONA.md: %s", e)

    draft = generate_persona_draft(self_content, persona_content)
    if not draft:
        logger.debug("run_persona_consolidation: LLM returned empty draft")
        return False

    write_persona_draft(draft)
    send_persona_diff_notification(draft, persona_content)

    summary = f"persona draft generated ({len(draft)} chars), notification sent"
    if on_summary:
        try:
            on_summary(summary)
        except Exception:
            pass

    return True
