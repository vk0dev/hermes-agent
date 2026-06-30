"""CLI subcommand: `hermes persona <subcommand>`.

Manages the PERSONA.md draft lifecycle: approve, reject, status, diff.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional


def _cmd_approve(args) -> int:
    from agent.persona_draft import promote_persona_draft
    ok, msg = promote_persona_draft()
    if ok:
        print(f"PERSONA.md promoted from draft: {msg}")
        return 0
    else:
        print(f"Promotion failed: {msg}", file=sys.stderr)
        return 1


def _cmd_reject(args) -> int:
    from agent.persona_draft import reject_persona_draft
    ok, msg = reject_persona_draft()
    if ok:
        print(f"Draft rejected: {msg}")
        return 0
    else:
        print(f"Reject failed: {msg}", file=sys.stderr)
        return 1


def _cmd_status(args) -> int:
    from agent.persona_draft import get_draft_status, _draft_path, _live_path
    status = get_draft_status()
    if status["exists"]:
        print(f"Draft: {status['path']} ({status['size']} bytes)")
        print("Approve:  hermes persona approve")
        print("Reject:   hermes persona reject")
        print("Preview:  hermes persona diff")
    else:
        print("No pending PERSONA.md draft.")
    live = _live_path()
    if live.exists():
        print(f"Live PERSONA.md: {live} ({live.stat().st_size} bytes)")
    return 0


def _cmd_diff(args) -> int:
    from agent.persona_draft import get_draft_diff
    diff = get_draft_diff()
    if diff:
        print(diff)
    else:
        print("No draft or no diff to show.")
    return 0


def register_cli(parent: argparse.ArgumentParser) -> None:
    parent.set_defaults(func=lambda a: (parent.print_help(), 0)[1])
    subs = parent.add_subparsers(dest="persona_action")
    subs.add_parser("approve", help="Promote PERSONA.md.draft to live PERSONA.md").set_defaults(func=_cmd_approve)
    subs.add_parser("reject", help="Delete PERSONA.md.draft without promoting").set_defaults(func=_cmd_reject)
    subs.add_parser("status", help="Show draft status").set_defaults(func=_cmd_status)
    subs.add_parser("diff", help="Show diff between draft and live PERSONA.md").set_defaults(func=_cmd_diff)
