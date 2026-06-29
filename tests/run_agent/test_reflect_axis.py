"""Tests for B1.2 reflect & distill review axis.

Verifies:
- spawn_background_review_thread dispatches reflect prompt independently.
- _spawn_background_review spawns a bg-reflect thread for review_reflect=True.
- record_fact handler calls Honcho create_conclusion via thread-local provider.
- append_reflection handler writes a first-person SELF.md line.
- Guards: second-person rejected, honcho unavailable gracefully skipped.
- Regression: memory/skills dispatch unchanged.
"""

from __future__ import annotations

import json
import re

import run_agent as run_agent_module
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Minimal bare agent for dispatch tests (no real AIAgent init)
# ---------------------------------------------------------------------------

def _bare_agent():
    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "telegram"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = "test-session"
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_manager = None
    agent._cached_system_prompt = "test-cached-system-prompt"
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_args, **_kwargs: None
    return agent


class CapturingThread:
    """Records thread construction; start() is a no-op."""
    def __init__(self, *, target, daemon=None, name=None):
        self.target = target
        self.name = name

    def start(self):
        pass


# ---------------------------------------------------------------------------
# Dispatch / selection tests
# ---------------------------------------------------------------------------

class TestDispatchSelection:
    def test_reflect_flag_returns_reflect_prompt(self):
        from agent.background_review import (
            spawn_background_review_thread,
            _REFLECT_DISTILL_PROMPT,
        )
        agent = _bare_agent()
        snap = [{"role": "user", "content": "hello"}]
        _target, prompt = spawn_background_review_thread(
            agent, snap, review_reflect=True
        )
        assert prompt == _REFLECT_DISTILL_PROMPT

    def test_memory_flag_returns_memory_prompt(self):
        from agent.background_review import spawn_background_review_thread
        agent = _bare_agent()
        _, prompt = spawn_background_review_thread(agent, [], review_memory=True)
        # spawn_background_review_thread respects per-agent prompt overrides
        assert prompt == agent._MEMORY_REVIEW_PROMPT

    def test_skills_flag_returns_skills_prompt(self):
        from agent.background_review import spawn_background_review_thread
        agent = _bare_agent()
        _, prompt = spawn_background_review_thread(agent, [], review_skills=True)
        assert prompt == agent._SKILL_REVIEW_PROMPT

    def test_memory_and_skills_flags_returns_combined_prompt(self):
        from agent.background_review import spawn_background_review_thread
        agent = _bare_agent()
        _, prompt = spawn_background_review_thread(
            agent, [], review_memory=True, review_skills=True
        )
        assert prompt == agent._COMBINED_REVIEW_PROMPT


# ---------------------------------------------------------------------------
# _spawn_background_review thread spawning
# ---------------------------------------------------------------------------

class TestSpawnBackgroundReview:
    def test_reflect_spawns_bg_reflect_thread(self, monkeypatch):
        spawned = []

        class RecordingThread:
            def __init__(self, *, target, daemon=None, name=None):
                spawned.append(name)
            def start(self):
                pass

        monkeypatch.setattr(run_agent_module.threading, "Thread", RecordingThread)
        agent = _bare_agent()
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[],
            review_reflect=True,
        )
        assert "bg-reflect" in spawned

    def test_memory_spawns_bg_review_thread(self, monkeypatch):
        spawned = []

        class RecordingThread:
            def __init__(self, *, target, daemon=None, name=None):
                spawned.append(name)
            def start(self):
                pass

        monkeypatch.setattr(run_agent_module.threading, "Thread", RecordingThread)
        agent = _bare_agent()
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[],
            review_memory=True,
        )
        assert "bg-review" in spawned
        assert "bg-reflect" not in spawned

    def test_reflect_and_memory_spawn_two_threads(self, monkeypatch):
        spawned = []

        class RecordingThread:
            def __init__(self, *, target, daemon=None, name=None):
                spawned.append(name)
            def start(self):
                pass

        monkeypatch.setattr(run_agent_module.threading, "Thread", RecordingThread)
        agent = _bare_agent()
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[],
            review_memory=True,
            review_reflect=True,
        )
        assert "bg-review" in spawned
        assert "bg-reflect" in spawned
        assert len(spawned) == 2


# ---------------------------------------------------------------------------
# record_fact handler — Honcho path
# ---------------------------------------------------------------------------

class TestRecordFact:
    def test_calls_create_conclusion_with_fact(self):
        from unittest.mock import MagicMock
        from tools.reflect_tool import (
            set_reflect_honcho_provider,
            clear_reflect_honcho_provider,
            _handle_record_fact,
        )

        mock_manager = MagicMock()
        mock_provider = MagicMock()
        mock_provider._manager = mock_manager
        mock_provider._session_key = "test-session-key"

        set_reflect_honcho_provider(mock_provider)
        try:
            result = _handle_record_fact(
                {"fact": "colleague Alex birthday is 2026-03-03"}
            )
        finally:
            clear_reflect_honcho_provider()

        data = json.loads(result)
        assert data["success"] is True
        mock_manager.create_conclusion.assert_called_once_with(
            "test-session-key",
            "colleague Alex birthday is 2026-03-03",
            peer="user",
        )

    def test_calls_create_conclusion_with_custom_peer(self):
        from unittest.mock import MagicMock
        from tools.reflect_tool import (
            set_reflect_honcho_provider,
            clear_reflect_honcho_provider,
            _handle_record_fact,
        )

        mock_manager = MagicMock()
        mock_provider = MagicMock()
        mock_provider._manager = mock_manager
        mock_provider._session_key = "sess-abc"

        set_reflect_honcho_provider(mock_provider)
        try:
            result = _handle_record_fact(
                {"fact": "project uses Python 3.12", "peer": "assistant"}
            )
        finally:
            clear_reflect_honcho_provider()

        data = json.loads(result)
        assert data["success"] is True
        mock_manager.create_conclusion.assert_called_once_with(
            "sess-abc", "project uses Python 3.12", peer="assistant"
        )

    def test_honcho_unavailable_returns_skipped(self):
        from tools.reflect_tool import clear_reflect_honcho_provider, _handle_record_fact

        clear_reflect_honcho_provider()
        result = _handle_record_fact({"fact": "some interesting fact"})
        data = json.loads(result)
        assert data["success"] is False
        assert data.get("skipped") == "honcho_unavailable"

    def test_honcho_no_manager_returns_skipped(self):
        from unittest.mock import MagicMock
        from tools.reflect_tool import (
            set_reflect_honcho_provider,
            clear_reflect_honcho_provider,
            _handle_record_fact,
        )

        mock_provider = MagicMock()
        mock_provider._manager = None
        mock_provider._session_key = "sess"

        set_reflect_honcho_provider(mock_provider)
        try:
            result = _handle_record_fact({"fact": "some fact"})
        finally:
            clear_reflect_honcho_provider()

        data = json.loads(result)
        assert data["success"] is False
        assert "skipped" in data

    def test_empty_fact_returns_error(self):
        from tools.reflect_tool import _handle_record_fact

        result = _handle_record_fact({"fact": ""})
        data = json.loads(result)
        assert data["success"] is False
        assert "error" in data

    def test_create_conclusion_exception_returns_skipped(self):
        from unittest.mock import MagicMock
        from tools.reflect_tool import (
            set_reflect_honcho_provider,
            clear_reflect_honcho_provider,
            _handle_record_fact,
        )

        mock_manager = MagicMock()
        mock_manager.create_conclusion.side_effect = RuntimeError("network error")
        mock_provider = MagicMock()
        mock_provider._manager = mock_manager
        mock_provider._session_key = "sess"

        set_reflect_honcho_provider(mock_provider)
        try:
            result = _handle_record_fact({"fact": "some fact"})
        finally:
            clear_reflect_honcho_provider()

        data = json.loads(result)
        assert data["success"] is False
        assert data.get("skipped") == "honcho_error"


# ---------------------------------------------------------------------------
# append_reflection handler — SELF.md path
# ---------------------------------------------------------------------------

class TestAppendReflectionTool:
    def test_writes_first_person_line_to_self_md(self):
        from tools.reflect_tool import _handle_append_reflection
        from hermes_constants import get_hermes_home

        result = _handle_append_reflection(
            {"reflection": "I learned to keep replies concise."}
        )
        data = json.loads(result)
        assert data["success"] is True

        content = (get_hermes_home() / "memories" / "SELF.md").read_text()
        assert re.search(
            r"\d{4}-\d{2}-\d{2} I learned to keep replies concise\.", content
        )

    def test_rejects_second_person(self):
        from tools.reflect_tool import _handle_append_reflection
        from hermes_constants import get_hermes_home

        result = _handle_append_reflection({"reflection": "You are terse."})
        data = json.loads(result)
        assert data["success"] is False
        assert not (get_hermes_home() / "memories" / "SELF.md").exists()

    def test_rejects_second_person_variant(self):
        from tools.reflect_tool import _handle_append_reflection
        from hermes_constants import get_hermes_home

        result = _handle_append_reflection({"reflection": "You're always right."})
        data = json.loads(result)
        assert data["success"] is False
        assert not (get_hermes_home() / "memories" / "SELF.md").exists()

    def test_rejects_empty_string(self):
        from tools.reflect_tool import _handle_append_reflection

        result = _handle_append_reflection({"reflection": ""})
        data = json.loads(result)
        assert data["success"] is False

    def test_multiple_entries_accumulate(self):
        from tools.reflect_tool import _handle_append_reflection
        from agent.self_journal import read_self_tail

        _handle_append_reflection({"reflection": "I value precision."})
        _handle_append_reflection({"reflection": "I prefer brevity in responses."})

        tail = read_self_tail()
        assert "I value precision." in tail
        assert "I prefer brevity in responses." in tail
