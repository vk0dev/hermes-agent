"""Tests for the reflect+distill axis (W5.2, t_b8ffee70) in background_review.

Covers:
1. Dispatch: spawn_background_review_thread selects _REFLECT_DISTILL_PROMPT when
   review_reflect=True alone; appends to existing prompt when combined with memory/skills.
2. review_reflect=False: prompt and behaviour are identical to pre-W5.2.
3. Whitelist: reflect adds honcho_conclude + self_reflect; non-reflect does NOT.
4. ReflectMemoryShim: has_tool, get_all_tool_schemas, handle_tool_call for both
   tools; lifecycle hooks are no-ops.
5. Fake integration: reflect fork calls honcho_conclude (→ fake parent mm) and
   self_reflect (→ real append_reflection in tmp HERMES_HOME).
6. Cadence (turn_finalizer): _turns_since_reflect increments each turn, fires on
   interval, resets, does NOT fire on intermediate turns.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_stub():
    """Minimal agent-like object for spawn_background_review_thread tests."""
    agent = MagicMock()
    agent._MEMORY_REVIEW_PROMPT = "memory prompt"
    agent._SKILL_REVIEW_PROMPT = "skills prompt"
    agent._COMBINED_REVIEW_PROMPT = "combined prompt"
    agent._REFLECT_DISTILL_PROMPT = "reflect prompt"
    agent._memory_manager = None
    return agent


# ---------------------------------------------------------------------------
# 1. Dispatch — prompt selection
# ---------------------------------------------------------------------------

def test_reflect_only_uses_reflect_distill_prompt():
    from agent.background_review import spawn_background_review_thread, _REFLECT_DISTILL_PROMPT

    agent = _make_agent_stub()
    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=False, review_skills=False, review_reflect=True
    )
    assert "REFLECT_DISTILL_PROMPT" not in prompt  # shouldn't contain literal name
    # The agent stub overrides the prompt via attribute; actual constant used when no attr
    assert prompt == "reflect prompt"


def test_reflect_only_uses_module_constant_when_no_agent_attr():
    from agent.background_review import spawn_background_review_thread, _REFLECT_DISTILL_PROMPT

    agent = _make_agent_stub()
    del agent._REFLECT_DISTILL_PROMPT  # remove per-agent override
    _target, prompt = spawn_background_review_thread(
        agent, [], review_reflect=True
    )
    assert prompt == _REFLECT_DISTILL_PROMPT


def test_reflect_combined_with_memory_appends_to_base():
    from agent.background_review import spawn_background_review_thread

    agent = _make_agent_stub()
    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_reflect=True
    )
    assert "memory prompt" in prompt
    assert "reflect prompt" in prompt


def test_reflect_combined_with_skills_appends_to_base():
    from agent.background_review import spawn_background_review_thread

    agent = _make_agent_stub()
    _target, prompt = spawn_background_review_thread(
        agent, [], review_skills=True, review_reflect=True
    )
    assert "skills prompt" in prompt
    assert "reflect prompt" in prompt


def test_reflect_combined_with_memory_and_skills():
    from agent.background_review import spawn_background_review_thread

    agent = _make_agent_stub()
    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=True, review_reflect=True
    )
    assert "combined prompt" in prompt
    assert "reflect prompt" in prompt


# ---------------------------------------------------------------------------
# 2. review_reflect=False — prompt unchanged (snapshot behaviour)
# ---------------------------------------------------------------------------

def test_reflect_false_memory_only_unchanged():
    from agent.background_review import spawn_background_review_thread

    agent = _make_agent_stub()
    _target, prompt = spawn_background_review_thread(agent, [], review_memory=True)
    assert prompt == "memory prompt"


def test_reflect_false_skills_only_unchanged():
    from agent.background_review import spawn_background_review_thread

    agent = _make_agent_stub()
    _target, prompt = spawn_background_review_thread(agent, [], review_skills=True)
    assert prompt == "skills prompt"


def test_reflect_false_combined_unchanged():
    from agent.background_review import spawn_background_review_thread

    agent = _make_agent_stub()
    _target, prompt = spawn_background_review_thread(
        agent, [], review_memory=True, review_skills=True
    )
    assert prompt == "combined prompt"


# ---------------------------------------------------------------------------
# 3. ReflectMemoryShim — unit tests
# ---------------------------------------------------------------------------

class _FakeParentMM:
    """Records honcho_conclude calls for assertion."""

    def __init__(self):
        self.calls: list = []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        self.calls.append((tool_name, args))
        conclusion = args.get("conclusion", "")
        peer = args.get("peer", "user")
        return json.dumps({"result": f"Conclusion saved for {peer}: {conclusion}"})


def test_shim_has_tool():
    from agent.background_review import ReflectMemoryShim

    shim = ReflectMemoryShim(parent_memory_manager=None, parent_agent=None)
    assert shim.has_tool("honcho_conclude")
    assert shim.has_tool("self_reflect")
    assert not shim.has_tool("memory")
    assert not shim.has_tool("skill_manage")
    assert not shim.has_tool("unknown_tool")


def test_shim_get_all_tool_schemas():
    from agent.background_review import ReflectMemoryShim

    shim = ReflectMemoryShim(parent_memory_manager=None, parent_agent=None)
    schemas = shim.get_all_tool_schemas()
    names = {s["name"] for s in schemas}
    assert "honcho_conclude" in names
    assert "self_reflect" in names


def test_shim_honcho_conclude_delegates_to_parent():
    from agent.background_review import ReflectMemoryShim

    parent_mm = _FakeParentMM()
    shim = ReflectMemoryShim(parent_memory_manager=parent_mm, parent_agent=None)
    result_raw = shim.handle_tool_call(
        "honcho_conclude", {"conclusion": "User birthday is April 1", "peer": "user"}
    )
    result = json.loads(result_raw)
    assert result["success"] is True
    assert "April 1" in result["message"] or "Conclusion saved" in result["message"]
    assert parent_mm.calls == [
        ("honcho_conclude", {"conclusion": "User birthday is April 1", "peer": "user"})
    ]


def test_shim_honcho_conclude_no_parent_graceful():
    from agent.background_review import ReflectMemoryShim

    shim = ReflectMemoryShim(parent_memory_manager=None, parent_agent=None)
    result = json.loads(shim.handle_tool_call("honcho_conclude", {"conclusion": "fact"}))
    assert result["success"] is False
    assert "No honcho provider" in result["message"]


def test_shim_self_reflect_writes_to_self_md(tmp_path):
    from agent.background_review import ReflectMemoryShim

    shim = ReflectMemoryShim(parent_memory_manager=None, parent_agent=None)

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        result = json.loads(
            shim.handle_tool_call("self_reflect", {"line": "I prefer concise responses."})
        )

    assert result["success"] is True
    assert "created" in result["message"].lower() or "Reflection" in result["message"]

    self_path = tmp_path / "memories" / "SELF.md"
    assert self_path.exists()
    content = self_path.read_text()
    assert "I prefer concise responses." in content


def test_shim_self_reflect_rejects_second_person(tmp_path):
    from agent.background_review import ReflectMemoryShim

    shim = ReflectMemoryShim(parent_memory_manager=None, parent_agent=None)
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        result = json.loads(
            shim.handle_tool_call("self_reflect", {"line": "You are very helpful."})
        )
    assert result["success"] is False


def test_shim_lifecycle_hooks_are_noop():
    from agent.background_review import ReflectMemoryShim

    shim = ReflectMemoryShim(parent_memory_manager=None, parent_agent=None)
    shim.on_turn_start(1, "hello")
    shim.prefetch_all()
    shim.queue_prefetch_all()
    shim.sync_all()
    shim.on_session_end([])
    shim.shutdown_all()
    shim.flush_all()
    shim.shutdown()
    shim.initialize_all()


# ---------------------------------------------------------------------------
# 4. Whitelist — reflect adds honcho_conclude + self_reflect
# ---------------------------------------------------------------------------

def test_whitelist_contains_reflect_tools_when_review_reflect_true(monkeypatch):
    """When review_reflect=True the thread-whitelist must include both reflect tools."""
    import agent.background_review as bg

    recorded_whitelist: set = set()

    def _fake_set_thread_tool_whitelist(wl, **kwargs):
        recorded_whitelist.update(wl)

    def _fake_clear_thread_tool_whitelist():
        pass

    def _fake_get_tool_definitions(**kwargs):
        return []

    def _fake_inject(agent):
        return 0

    monkeypatch.setattr(
        "hermes_cli.plugins.set_thread_tool_whitelist", _fake_set_thread_tool_whitelist
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.clear_thread_tool_whitelist", _fake_clear_thread_tool_whitelist
    )
    monkeypatch.setattr("model_tools.get_tool_definitions", _fake_get_tool_definitions)
    monkeypatch.setattr("agent.memory_manager.inject_memory_provider_tools", _fake_inject)

    # Patch AIAgent constructor so we never actually spawn a real agent
    class _FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []
            self.model = "x"
            self.platform = "x"
            self.provider = "x"
            self.base_url = ""
            self.api_key = ""
            self.api_mode = ""
            self.session_id = "r"
            self.enabled_toolsets = ["memory", "skills"]
            self.tools = []
            self.valid_tool_names = set()

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    import run_agent as _ra
    monkeypatch.setattr(_ra, "AIAgent", _FakeReviewAgent)

    parent_agent = MagicMock()
    parent_agent._current_main_runtime.return_value = {
        "api_mode": "openai", "base_url": "", "api_key": ""
    }
    parent_agent._memory_manager = None
    parent_agent.model = "m"
    parent_agent.platform = "p"
    parent_agent.provider = "openai"
    parent_agent.session_id = "s"
    parent_agent.session_start = None
    parent_agent._cached_system_prompt = ""
    parent_agent._memory_store = None
    parent_agent._memory_enabled = False
    parent_agent._user_profile_enabled = False
    parent_agent.enabled_toolsets = ["memory", "skills"]
    parent_agent.disabled_toolsets = []
    parent_agent._credential_pool = None
    parent_agent._safe_print = lambda *a, **k: None
    parent_agent.background_review_callback = None
    parent_agent.memory_notifications = "on"

    monkeypatch.setattr(_ra.threading, "Thread", type(
        "_Immediate", (), {
            "__init__": lambda self, *, target, daemon=None, name=None: setattr(self, "_t", target),
            "start": lambda self: self._t(),
        }
    ))

    bg._run_review_in_thread(parent_agent, [], "review prompt", review_reflect=True)

    assert "honcho_conclude" in recorded_whitelist
    assert "self_reflect" in recorded_whitelist


def test_whitelist_does_not_contain_reflect_tools_when_false(monkeypatch):
    """When review_reflect=False the whitelist must NOT contain reflect tools."""
    import agent.background_review as bg

    recorded_whitelist: set = set()

    def _fake_set_thread_tool_whitelist(wl, **kwargs):
        recorded_whitelist.update(wl)

    def _fake_clear_thread_tool_whitelist():
        pass

    def _fake_get_tool_definitions(**kwargs):
        return []

    monkeypatch.setattr(
        "hermes_cli.plugins.set_thread_tool_whitelist", _fake_set_thread_tool_whitelist
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.clear_thread_tool_whitelist", _fake_clear_thread_tool_whitelist
    )
    monkeypatch.setattr("model_tools.get_tool_definitions", _fake_get_tool_definitions)

    class _FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []
            self.model = "x"
            self.platform = "x"
            self.provider = "x"
            self.base_url = ""
            self.api_key = ""
            self.api_mode = ""
            self.session_id = "r"
            self.enabled_toolsets = ["memory", "skills"]
            self.tools = []
            self.valid_tool_names = set()

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    import run_agent as _ra
    monkeypatch.setattr(_ra, "AIAgent", _FakeReviewAgent)

    parent_agent = MagicMock()
    parent_agent._current_main_runtime.return_value = {
        "api_mode": "openai", "base_url": "", "api_key": ""
    }
    parent_agent._memory_manager = None
    parent_agent.model = "m"
    parent_agent.platform = "p"
    parent_agent.provider = "openai"
    parent_agent.session_id = "s"
    parent_agent.session_start = None
    parent_agent._cached_system_prompt = ""
    parent_agent._memory_store = None
    parent_agent._memory_enabled = False
    parent_agent._user_profile_enabled = False
    parent_agent.enabled_toolsets = ["memory", "skills"]
    parent_agent.disabled_toolsets = []
    parent_agent._credential_pool = None
    parent_agent._safe_print = lambda *a, **k: None
    parent_agent.background_review_callback = None
    parent_agent.memory_notifications = "on"

    monkeypatch.setattr(_ra.threading, "Thread", type(
        "_Immediate", (), {
            "__init__": lambda self, *, target, daemon=None, name=None: setattr(self, "_t", target),
            "start": lambda self: self._t(),
        }
    ))

    bg._run_review_in_thread(parent_agent, [], "review prompt", review_reflect=False)

    assert "honcho_conclude" not in recorded_whitelist
    assert "self_reflect" not in recorded_whitelist


# ---------------------------------------------------------------------------
# 5. Cadence — turn_finalizer increments and fires on interval
# ---------------------------------------------------------------------------

def _make_finalizer_agent():
    """Minimal agent for turn_finalizer cadence tests."""
    agent = MagicMock()
    agent._reflect_nudge_interval = 3
    agent._turns_since_reflect = 0
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = set()
    agent._safe_print = lambda *a, **k: None
    return agent


def test_reflect_fires_on_interval():
    """_should_review_reflect becomes True exactly when counter reaches interval."""
    from agent import turn_finalizer

    agent = _make_finalizer_agent()
    spawn_calls: list = []

    agent._sync_external_memory_for_turn = MagicMock()
    agent._spawn_background_review = lambda **kw: spawn_calls.append(kw)

    # Simulate finalize_turn's reflect logic directly (we test the snippet
    # since finalize_turn has many other dependencies).
    def _simulate_reflect_check(agent):
        _reflect_interval = getattr(agent, "_reflect_nudge_interval", 0)
        _should_review_reflect = False
        if _reflect_interval > 0:
            agent._turns_since_reflect = getattr(agent, "_turns_since_reflect", 0) + 1
            if agent._turns_since_reflect >= _reflect_interval:
                _should_review_reflect = True
                agent._turns_since_reflect = 0
        return _should_review_reflect

    turn1 = _simulate_reflect_check(agent)
    assert not turn1
    assert agent._turns_since_reflect == 1

    turn2 = _simulate_reflect_check(agent)
    assert not turn2
    assert agent._turns_since_reflect == 2

    turn3 = _simulate_reflect_check(agent)
    assert turn3  # interval=3, should fire
    assert agent._turns_since_reflect == 0  # reset

    turn4 = _simulate_reflect_check(agent)
    assert not turn4
    assert agent._turns_since_reflect == 1


def test_reflect_does_not_fire_when_interval_zero():
    def _simulate(agent):
        _reflect_interval = getattr(agent, "_reflect_nudge_interval", 0)
        _should = False
        if _reflect_interval > 0:
            agent._turns_since_reflect = getattr(agent, "_turns_since_reflect", 0) + 1
            if agent._turns_since_reflect >= _reflect_interval:
                _should = True
                agent._turns_since_reflect = 0
        return _should

    agent = MagicMock()
    agent._reflect_nudge_interval = 0
    agent._turns_since_reflect = 100  # even if counter is high, no fire

    for _ in range(5):
        assert not _simulate(agent)


# ---------------------------------------------------------------------------
# 6. summarize_background_review_actions includes reflect actions
# ---------------------------------------------------------------------------

def test_summarizer_surfaces_honcho_conclude_action():
    from agent.background_review import summarize_background_review_actions

    tool_call_id = "tc-conclude-1"
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": tool_call_id,
                "function": {
                    "name": "honcho_conclude",
                    "arguments": json.dumps({"conclusion": "User birthday April 1", "peer": "user"}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"success": True, "message": "Durable fact created: User birthday April 1"}),
        },
    ]

    actions = summarize_background_review_actions(messages, [], notification_mode="on")
    assert any("created" in a.lower() or "fact" in a.lower() for a in actions), (
        f"Expected honcho_conclude action in summary, got: {actions}"
    )


def test_summarizer_surfaces_self_reflect_action():
    from agent.background_review import summarize_background_review_actions

    tool_call_id = "tc-reflect-1"
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": tool_call_id,
                "function": {
                    "name": "self_reflect",
                    "arguments": json.dumps({"line": "I prefer concise responses."}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"success": True, "message": "Reflection created: I prefer concise responses."}),
        },
    ]

    actions = summarize_background_review_actions(messages, [], notification_mode="on")
    assert any("created" in a.lower() or "reflection" in a.lower() for a in actions), (
        f"Expected self_reflect action in summary, got: {actions}"
    )


# ---------------------------------------------------------------------------
# 7. Session-end trigger — maybe_spawn_session_end_reflect
# ---------------------------------------------------------------------------

def _make_session_end_agent(*, nudge_interval=10, turns_since=4):
    agent = MagicMock()
    agent._reflect_nudge_interval = nudge_interval
    agent._turns_since_reflect = turns_since
    spawn_calls = []
    agent._spawn_background_review = lambda **kw: spawn_calls.append(kw)
    agent._spawn_calls = spawn_calls
    return agent


def test_session_end_reflect_fires_when_turns_pending():
    from agent.background_review import maybe_spawn_session_end_reflect

    agent = _make_session_end_agent(nudge_interval=10, turns_since=4)
    result = maybe_spawn_session_end_reflect(agent, [{"role": "user", "content": "hi"}])

    assert result is True
    assert len(agent._spawn_calls) == 1
    assert agent._spawn_calls[0]["review_reflect"] is True
    assert agent._turns_since_reflect == 0


def test_session_end_reflect_skips_when_no_pending_turns():
    from agent.background_review import maybe_spawn_session_end_reflect

    agent = _make_session_end_agent(nudge_interval=10, turns_since=0)
    result = maybe_spawn_session_end_reflect(agent, [{"role": "user", "content": "hi"}])

    assert result is False
    assert len(agent._spawn_calls) == 0


def test_session_end_reflect_skips_when_interval_zero():
    from agent.background_review import maybe_spawn_session_end_reflect

    agent = _make_session_end_agent(nudge_interval=0, turns_since=99)
    result = maybe_spawn_session_end_reflect(agent, [{"role": "user", "content": "hi"}])

    assert result is False
    assert len(agent._spawn_calls) == 0


def test_session_end_reflect_skips_on_empty_snapshot():
    from agent.background_review import maybe_spawn_session_end_reflect

    agent = _make_session_end_agent(nudge_interval=10, turns_since=5)
    result = maybe_spawn_session_end_reflect(agent, [])

    assert result is False
    assert len(agent._spawn_calls) == 0


def test_session_end_reflect_swallows_spawn_error():
    from agent.background_review import maybe_spawn_session_end_reflect

    agent = MagicMock()
    agent._reflect_nudge_interval = 10
    agent._turns_since_reflect = 5
    agent._spawn_background_review = MagicMock(side_effect=RuntimeError("boom"))

    result = maybe_spawn_session_end_reflect(agent, [{"role": "user", "content": "hi"}])

    assert result is False  # error swallowed, no raise


def test_session_end_reflect_via_commit_memory_session(monkeypatch):
    """commit_memory_session calls maybe_spawn_session_end_reflect best-effort."""
    import run_agent as _ra
    from unittest.mock import MagicMock

    spawn_calls = []

    agent = MagicMock(spec=_ra.AIAgent)
    agent._reflect_nudge_interval = 5
    agent._turns_since_reflect = 3
    agent._memory_manager = None
    agent.context_compressor = None
    agent._spawn_background_review = lambda **kw: spawn_calls.append(kw)

    _ra.AIAgent.commit_memory_session(agent, [{"role": "user", "content": "x"}])

    assert len(spawn_calls) == 1
    assert spawn_calls[0]["review_reflect"] is True


# ---------------------------------------------------------------------------
# 8. CLI _run_cleanup trigger — session-end reflect on true exit
# ---------------------------------------------------------------------------

def test_cli_run_cleanup_fires_session_end_reflect(monkeypatch):
    """_run_cleanup calls maybe_spawn_session_end_reflect when turns are pending."""
    import cli as _cli
    from unittest.mock import MagicMock

    spawn_calls = []
    agent = MagicMock()
    agent._reflect_nudge_interval = 5
    agent._turns_since_reflect = 3
    agent._session_messages = [{"role": "user", "content": "hello"}]
    agent._spawn_background_review = lambda **kw: spawn_calls.append(kw)
    agent.shutdown_memory_provider = MagicMock()

    monkeypatch.setattr(_cli, "_active_agent_ref", agent)
    monkeypatch.setattr(_cli, "_cleanup_done", False)
    monkeypatch.setattr(_cli, "_reset_terminal_input_modes_on_exit", lambda: None)
    monkeypatch.setattr(_cli, "_cleanup_all_terminals", lambda: None)
    monkeypatch.setattr(_cli, "_cleanup_all_browsers", lambda: None)

    _cli._run_cleanup(notify_session_finalize=False)

    assert len(spawn_calls) == 1
    assert spawn_calls[0]["review_reflect"] is True
    assert agent._turns_since_reflect == 0


def test_cli_run_cleanup_skips_reflect_when_no_pending_turns(monkeypatch):
    """_run_cleanup does NOT fire reflect when _turns_since_reflect == 0 (de-dup)."""
    import cli as _cli
    from unittest.mock import MagicMock

    spawn_calls = []
    agent = MagicMock()
    agent._reflect_nudge_interval = 5
    agent._turns_since_reflect = 0  # interval just fired — de-dup guard
    agent._session_messages = [{"role": "user", "content": "hello"}]
    agent._spawn_background_review = lambda **kw: spawn_calls.append(kw)
    agent.shutdown_memory_provider = MagicMock()

    monkeypatch.setattr(_cli, "_active_agent_ref", agent)
    monkeypatch.setattr(_cli, "_cleanup_done", False)
    monkeypatch.setattr(_cli, "_reset_terminal_input_modes_on_exit", lambda: None)
    monkeypatch.setattr(_cli, "_cleanup_all_terminals", lambda: None)
    monkeypatch.setattr(_cli, "_cleanup_all_browsers", lambda: None)

    _cli._run_cleanup(notify_session_finalize=False)

    assert len(spawn_calls) == 0


# ---------------------------------------------------------------------------
# 9. Gateway expiry watcher trigger — maybe_flush_reflect_on_idle
# ---------------------------------------------------------------------------

def test_maybe_flush_reflect_on_idle_delegates_to_session_end():
    """maybe_flush_reflect_on_idle is a named alias for maybe_spawn_session_end_reflect."""
    from agent.background_review import maybe_flush_reflect_on_idle, maybe_spawn_session_end_reflect
    from unittest.mock import MagicMock, patch

    agent = MagicMock()
    agent._reflect_nudge_interval = 5
    agent._turns_since_reflect = 3
    spawn_calls = []
    agent._spawn_background_review = lambda **kw: spawn_calls.append(kw)

    result = maybe_flush_reflect_on_idle(agent, [{"role": "user", "content": "hi"}])

    assert result is True
    assert len(spawn_calls) == 1
    assert agent._turns_since_reflect == 0


def test_gateway_expiry_reflect_best_effort_on_error():
    """maybe_flush_reflect_on_idle errors are swallowed (best-effort contract)."""
    from agent.background_review import maybe_flush_reflect_on_idle
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent._reflect_nudge_interval = 5
    agent._turns_since_reflect = 3
    agent._spawn_background_review = MagicMock(side_effect=RuntimeError("reflect boom"))

    result = maybe_flush_reflect_on_idle(agent, [{"role": "user", "content": "hi"}])

    assert result is False  # error swallowed
