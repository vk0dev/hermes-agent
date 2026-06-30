"""Tests for agent/persona_draft.py — PERSONA.md draft pipeline.

All tests are fully offline:
  - send_message_tool is mocked to avoid real Telegram sends
  - _run_llm_review is mocked to avoid real LLM calls
  - load_split_soul_md is mocked in promotion tests
  - HERMES_HOME is redirected to a tmp dir by the autouse conftest fixture
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hermes_root() -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root()


def _write_self_md(content: str) -> None:
    from hermes_constants import get_hermes_home
    self_path = get_hermes_home() / "memories" / "SELF.md"
    self_path.parent.mkdir(parents=True, exist_ok=True)
    self_path.write_text(content, encoding="utf-8")


def _write_live_persona(content: str) -> None:
    live = _hermes_root() / "PERSONA.md"
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_text(content, encoding="utf-8")


def _write_draft(content: str) -> None:
    draft = _hermes_root() / "PERSONA.md.draft"
    draft.parent.mkdir(parents=True, exist_ok=True)
    draft.write_text(content, encoding="utf-8")


VALID_PERSONA = "I am Hermes. I prefer clarity over verbosity. I value autonomy."
INVALID_PERSONA_NO_IAM = "This agent assists users with various tasks efficiently."
INVALID_PERSONA_SECOND_PERSON = "You are Hermes. You prefer clear responses."


# ---------------------------------------------------------------------------
# TestWritePersonaDraft
# ---------------------------------------------------------------------------

class TestWritePersonaDraft:
    def test_writes_to_draft_not_live(self):
        from agent.persona_draft import write_persona_draft
        result = write_persona_draft("I am a draft.")
        assert result is True
        draft = _hermes_root() / "PERSONA.md.draft"
        live = _hermes_root() / "PERSONA.md"
        assert draft.exists()
        assert not live.exists()

    def test_draft_content_matches(self):
        from agent.persona_draft import write_persona_draft
        content = "I am the draft content. I prefer tests to pass."
        write_persona_draft(content)
        draft = _hermes_root() / "PERSONA.md.draft"
        assert draft.read_text(encoding="utf-8") == content

    def test_write_empty_string(self):
        """Empty string is allowed for the draft file."""
        from agent.persona_draft import write_persona_draft
        result = write_persona_draft("")
        assert result is True
        draft = _hermes_root() / "PERSONA.md.draft"
        assert draft.exists()
        assert draft.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# TestPromotePersonaDraft
# ---------------------------------------------------------------------------

class TestPromotePersonaDraft:
    def test_promote_valid_draft(self):
        _write_draft(VALID_PERSONA)
        with patch("agent.prompt_builder.load_split_soul_md", return_value=VALID_PERSONA):
            from agent.persona_draft import promote_persona_draft
            ok, msg = promote_persona_draft()
        assert ok is True
        assert "promoted" in msg
        live = _hermes_root() / "PERSONA.md"
        assert live.exists()
        assert live.read_text(encoding="utf-8") == VALID_PERSONA

    def test_promote_creates_backup(self):
        _write_live_persona(VALID_PERSONA)
        new_content = "I am Hermes updated. I value precision."
        _write_draft(new_content)
        with patch("agent.prompt_builder.load_split_soul_md", return_value=new_content):
            from agent.persona_draft import promote_persona_draft
            ok, msg = promote_persona_draft()
        assert ok is True
        bak = _hermes_root() / "PERSONA.md.bak"
        assert bak.exists()
        assert bak.read_text(encoding="utf-8") == VALID_PERSONA

    def test_promote_draft_deleted_after(self):
        _write_draft(VALID_PERSONA)
        with patch("agent.prompt_builder.load_split_soul_md", return_value=VALID_PERSONA):
            from agent.persona_draft import promote_persona_draft
            ok, _ = promote_persona_draft()
        assert ok is True
        draft = _hermes_root() / "PERSONA.md.draft"
        assert not draft.exists()

    def test_promote_second_person_rejected(self):
        _write_live_persona(VALID_PERSONA)
        _write_draft(INVALID_PERSONA_SECOND_PERSON)
        from agent.persona_draft import promote_persona_draft
        ok, msg = promote_persona_draft()
        assert ok is False
        assert "authoring" in msg
        # Live PERSONA.md unchanged
        live = _hermes_root() / "PERSONA.md"
        assert live.read_text(encoding="utf-8") == VALID_PERSONA

    def test_promote_missing_i_am_rejected(self):
        _write_live_persona(VALID_PERSONA)
        _write_draft(INVALID_PERSONA_NO_IAM)
        from agent.persona_draft import promote_persona_draft
        ok, msg = promote_persona_draft()
        assert ok is False
        assert "authoring" in msg
        live = _hermes_root() / "PERSONA.md"
        assert live.read_text(encoding="utf-8") == VALID_PERSONA

    def test_promote_no_draft(self):
        from agent.persona_draft import promote_persona_draft
        ok, msg = promote_persona_draft()
        assert ok is False
        assert "no draft" in msg.lower()

    def test_promote_soul_guard_triggers_rollback(self):
        """If load_split_soul_md returns a sentinel, rollback and return error."""
        original_content = VALID_PERSONA
        _write_live_persona(original_content)
        _write_draft("I am valid. I value goodness.")
        with patch("agent.prompt_builder.load_split_soul_md",
                   return_value="[SOUL-LOST: test sentinel]"):
            from agent.persona_draft import promote_persona_draft
            ok, msg = promote_persona_draft()
        assert ok is False
        assert "soul" in msg.lower() or "rolled back" in msg.lower()
        # Live PERSONA.md should be rolled back to original
        live = _hermes_root() / "PERSONA.md"
        assert live.read_text(encoding="utf-8") == original_content


# ---------------------------------------------------------------------------
# TestRejectPersonaDraft
# ---------------------------------------------------------------------------

class TestRejectPersonaDraft:
    def test_reject_deletes_draft(self):
        _write_draft("I am a draft to be rejected.")
        from agent.persona_draft import reject_persona_draft
        ok, msg = reject_persona_draft()
        assert ok is True
        draft = _hermes_root() / "PERSONA.md.draft"
        assert not draft.exists()

    def test_reject_no_draft(self):
        from agent.persona_draft import reject_persona_draft
        ok, msg = reject_persona_draft()
        assert ok is True
        assert "no draft" in msg.lower()


# ---------------------------------------------------------------------------
# TestGetDraftDiff
# ---------------------------------------------------------------------------

class TestGetDraftDiff:
    def test_no_draft_returns_empty(self):
        from agent.persona_draft import get_draft_diff
        assert get_draft_diff() == ""

    def test_draft_only_no_live(self):
        _write_draft("I am a new persona. I value simplicity.")
        from agent.persona_draft import get_draft_diff
        diff = get_draft_diff()
        # With no live, returns draft content as-is
        assert "I am a new persona" in diff

    def test_draft_and_live_unified_diff(self):
        _write_live_persona("I am Hermes. I value order.")
        _write_draft("I am Hermes. I value order and clarity.")
        from agent.persona_draft import get_draft_diff
        diff = get_draft_diff()
        assert diff  # has content
        assert "PERSONA.md" in diff  # unified diff header
        assert "clarity" in diff  # new content shows up


# ---------------------------------------------------------------------------
# TestSendPersonaDiffNotification
# ---------------------------------------------------------------------------

class TestSendPersonaDiffNotification:
    def test_calls_send_with_correct_target(self):
        from agent.persona_draft import send_persona_diff_notification
        captured = []

        def mock_send(args, **kw):
            captured.append(args)
            return '{"ok": true}'

        with patch("tools.send_message_tool.send_message_tool", mock_send):
            result = send_persona_diff_notification("I am a draft persona.", "")

        assert result is True
        assert len(captured) == 1
        assert captured[0]["target"] == "telegram:552296"

    def test_message_includes_approve_reject_instructions(self):
        from agent.persona_draft import send_persona_diff_notification
        messages = []

        def mock_send(args, **kw):
            messages.append(args.get("message", ""))
            return '{"ok": true}'

        with patch("tools.send_message_tool.send_message_tool", mock_send):
            send_persona_diff_notification("I am a draft.", "")

        assert messages
        msg = messages[0]
        assert "hermes persona approve" in msg
        assert "hermes persona reject" in msg

    def test_send_failure_returns_false_no_exception(self):
        from agent.persona_draft import send_persona_diff_notification

        def mock_send(args, **kw):
            raise RuntimeError("Network error")

        with patch("tools.send_message_tool.send_message_tool", mock_send):
            result = send_persona_diff_notification("I am a draft.", "")

        assert result is False


# ---------------------------------------------------------------------------
# TestRunPersonaConsolidation
# ---------------------------------------------------------------------------

class TestRunPersonaConsolidation:
    def test_empty_self_md_returns_false(self):
        # Don't write SELF.md — it doesn't exist
        from agent.persona_draft import run_persona_consolidation
        with patch("tools.send_message_tool.send_message_tool", MagicMock()):
            result = run_persona_consolidation()
        assert result is False
        draft = _hermes_root() / "PERSONA.md.draft"
        assert not draft.exists()

    def test_with_self_md_generates_draft_and_notifies(self):
        _write_self_md("2026-06-29 I notice I prefer concise responses over verbose ones.")

        mock_llm_result = {
            "final": "I am Hermes. I prefer concise responses. I value clarity.",
            "summary": "generated persona",
            "model": "test",
            "provider": "test",
            "tool_calls": [],
            "error": None,
        }

        send_calls = []

        def mock_send(args, **kw):
            send_calls.append(args)
            return '{"ok": true}'

        with patch("agent.curator._run_llm_review", return_value=mock_llm_result):
            with patch("tools.send_message_tool.send_message_tool", mock_send):
                from agent.persona_draft import run_persona_consolidation
                result = run_persona_consolidation()

        assert result is True
        draft = _hermes_root() / "PERSONA.md.draft"
        assert draft.exists()
        assert "I am Hermes" in draft.read_text(encoding="utf-8")
        assert len(send_calls) == 1

    def test_on_summary_callback_called(self):
        _write_self_md("2026-06-29 I learned something valuable.")

        mock_llm_result = {
            "final": "I am Hermes. I value learning.",
            "summary": "done",
            "model": "test",
            "provider": "test",
            "tool_calls": [],
            "error": None,
        }
        summaries = []

        with patch("agent.curator._run_llm_review", return_value=mock_llm_result):
            with patch("tools.send_message_tool.send_message_tool", MagicMock()):
                from agent.persona_draft import run_persona_consolidation
                result = run_persona_consolidation(on_summary=summaries.append)

        assert result is True
        assert summaries  # on_summary was called

    def test_llm_returning_empty_draft_returns_false(self):
        _write_self_md("2026-06-29 I learned something.")

        mock_llm_result = {
            "final": "",
            "summary": "empty",
            "model": "test",
            "provider": "test",
            "tool_calls": [],
            "error": None,
        }

        with patch("agent.curator._run_llm_review", return_value=mock_llm_result):
            with patch("tools.send_message_tool.send_message_tool", MagicMock()):
                from agent.persona_draft import run_persona_consolidation
                result = run_persona_consolidation()

        assert result is False
        draft = _hermes_root() / "PERSONA.md.draft"
        assert not draft.exists()


# ---------------------------------------------------------------------------
# TestIdleFlushReflect
# ---------------------------------------------------------------------------

def _make_agent(*, nudge_interval=10, turns_since=4):
    agent = MagicMock()
    agent._reflect_nudge_interval = nudge_interval
    agent._turns_since_reflect = turns_since
    spawn_calls = []
    agent._spawn_background_review = lambda **kw: spawn_calls.append(kw)
    agent._spawn_calls = spawn_calls
    return agent


class TestIdleFlushReflect:
    def test_fires_when_turns_pending(self):
        from agent.background_review import maybe_flush_reflect_on_idle
        agent = _make_agent(nudge_interval=10, turns_since=4)
        result = maybe_flush_reflect_on_idle(agent, [{"role": "user", "content": "hi"}])
        assert result is True
        assert len(agent._spawn_calls) == 1
        assert agent._spawn_calls[0]["review_reflect"] is True
        assert agent._turns_since_reflect == 0

    def test_returns_false_when_turns_zero(self):
        from agent.background_review import maybe_flush_reflect_on_idle
        agent = _make_agent(nudge_interval=10, turns_since=0)
        result = maybe_flush_reflect_on_idle(agent, [{"role": "user", "content": "hi"}])
        assert result is False
        assert len(agent._spawn_calls) == 0

    def test_returns_false_when_interval_zero(self):
        from agent.background_review import maybe_flush_reflect_on_idle
        agent = _make_agent(nudge_interval=0, turns_since=99)
        result = maybe_flush_reflect_on_idle(agent, [{"role": "user", "content": "hi"}])
        assert result is False
        assert len(agent._spawn_calls) == 0
