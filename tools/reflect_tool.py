"""Reflect & distill tools for the background reflect review axis (B1.2).

Two tools, both restricted to the ``reflect`` toolset:
  - ``append_reflection``: write a first-person observation to SELF.md via
    the existing :func:`agent.self_journal.append_reflection` guard.
  - ``record_fact``: store a durable world/person/project fact to Honcho
    via ``create_conclusion`` on the parent's live provider, reached through
    a thread-local handoff so the reflect fork (``skip_memory=True``) never
    rebuilds its own memory provider and stays isolated from ingestion.

Thread-local handoff pattern mirrors ``set_thread_tool_whitelist`` in
``hermes_cli.plugins``.  The parent sets the provider before spawning the
reflect thread; the handler picks it up; the parent clears it in ``finally``.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-local Honcho provider handoff
# ---------------------------------------------------------------------------

_reflect_ctx = threading.local()


def set_reflect_honcho_provider(provider: Any) -> None:
    """Set the Honcho provider for the current reflect-review thread."""
    _reflect_ctx.honcho = provider


def clear_reflect_honcho_provider() -> None:
    """Clear the Honcho provider for the current reflect-review thread."""
    _reflect_ctx.honcho = None


def _get_reflect_honcho_provider() -> Any:
    return getattr(_reflect_ctx, "honcho", None)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _handle_append_reflection(args: dict, **_kw) -> str:
    """Wrap agent.self_journal.append_reflection as a dispatchable tool."""
    reflection = (args.get("reflection") or "").strip()
    from agent.self_journal import append_reflection
    ok = append_reflection(reflection)
    return json.dumps({"success": ok})


def _handle_record_fact(args: dict, **_kw) -> str:
    """Write a durable fact to Honcho via the parent's live provider.

    Graceful no-op when the provider is unavailable (``skip_memory=True``
    on the reflect fork means the parent must handoff the provider via
    ``set_reflect_honcho_provider`` before calling ``run_conversation``).
    """
    fact = (args.get("fact") or "").strip()
    peer = args.get("peer") or "user"
    if not fact:
        return json.dumps({"success": False, "error": "fact is required"})
    provider = _get_reflect_honcho_provider()
    if provider is None:
        return json.dumps({"success": False, "skipped": "honcho_unavailable"})
    try:
        manager = getattr(provider, "_manager", None)
        session_key = getattr(provider, "_session_key", None)
        if manager is None or not session_key:
            return json.dumps({"success": False, "skipped": "honcho_unavailable"})
        manager.create_conclusion(session_key, fact, peer=peer)
        return json.dumps({"success": True})
    except Exception as exc:
        logger.debug("record_fact: create_conclusion failed: %s", exc)
        return json.dumps({"success": False, "skipped": "honcho_error", "detail": str(exc)})


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_APPEND_REFLECTION_SCHEMA = {
    "name": "append_reflection",
    "description": (
        "Append a first-person self-reflection to the SELF.md journal. "
        "Use this to record observations about how you work, patterns you notice, "
        "lessons learned about your own behaviour, or durable first-person insights. "
        "The entry is datestamped and subject to a FIFO cap of 10 entries. "
        "Second-person constructs ('You are …', 'You're …') are rejected by the guard. "
        "Write in the first person: 'I learned …', 'I tend to …', 'When X, I …'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reflection": {
                "type": "string",
                "description": (
                    "First-person reflection text. "
                    "Example: 'I learned to keep replies concise when context is short.'"
                ),
            },
        },
        "required": ["reflection"],
    },
}

_RECORD_FACT_SCHEMA = {
    "name": "record_fact",
    "description": (
        "Record a durable world, person, or project fact to long-term memory via "
        "Honcho conclusion. Use this for externally-verifiable facts about people, "
        "projects, or the world discovered during the conversation "
        "(e.g. 'colleague Alex birthday is 2026-03-03', "
        "'the team migrated to Python 3.12 in April 2026'). "
        "Do NOT use for first-person self-reflection — use append_reflection for that. "
        "Gracefully no-ops when Honcho is unavailable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": (
                    "The durable world/person/project fact to record. "
                    "Not first-person self-reflection — use append_reflection for that."
                ),
            },
            "peer": {
                "type": "string",
                "description": "Whose perspective this fact is from ('user' by default).",
            },
        },
        "required": ["fact"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="append_reflection",
    toolset="reflect",
    schema=_APPEND_REFLECTION_SCHEMA,
    handler=_handle_append_reflection,
)

registry.register(
    name="record_fact",
    toolset="reflect",
    schema=_RECORD_FACT_SCHEMA,
    handler=_handle_record_fact,
)
