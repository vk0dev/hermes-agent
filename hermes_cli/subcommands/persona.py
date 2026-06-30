"""``hermes persona`` subcommand parser."""

from __future__ import annotations

import logging


def build_persona_parser(subparsers) -> None:
    """Attach the ``persona`` subcommand to ``subparsers`` (B1.3 draft lifecycle)."""
    persona_parser = subparsers.add_parser(
        "persona",
        help="Manage PERSONA.md draft lifecycle (approve, reject, status, diff)",
    )
    try:
        from hermes_cli.persona import register_cli as _register_persona_cli

        _register_persona_cli(persona_parser)
    except Exception as _exc:
        logging.getLogger("hermes_cli.main").debug("persona CLI wiring failed: %s", _exc)
