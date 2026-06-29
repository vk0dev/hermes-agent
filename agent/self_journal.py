"""SELF.md journal for Hermes main agent self-reflection.

Stores first-person reflection entries in SELF.md under HERMES_HOME/memories/.
Entries are datestamped (YYYY-MM-DD) and capped at 10 via FIFO eviction.

Guard note: we intentionally do NOT reuse _validate_persona_authoring from
agent.prompt_builder.  That function requires ``\\bI am\\b`` — which would
reject valid short/non-English reflections like «понял, что предпочитаю краткость».
The acceptance requirement is narrower: only reject second-person authoring and
empty input.  Local equivalent logic fulfils the spec.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
from pathlib import Path

_log = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\b")
_SECOND_PERSON_RE = re.compile(r"\bYou are\b|\bYou're\b", re.IGNORECASE)
_MAX_ENTRIES = 10


def _self_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "memories" / "SELF.md"


def _validate_reflection_authoring(line: str) -> bool:
    """Return True if line passes first-person authoring guard.

    Rejects: empty/whitespace-only input, second-person directives («You are»,
    «You're»).  Does NOT require «I am» — see module docstring for rationale.
    """
    if not line.strip():
        return False
    if _SECOND_PERSON_RE.search(line):
        return False
    return True


def _parse(text: str) -> tuple[str, list[str]]:
    """Split SELF.md content into (header, reflections).

    header  — leading block of lines before the first datestamped line,
              preserved verbatim (includes #-headings, HTML comments, blanks).
    reflections — ordered list of datestamped reflection lines.
    Non-datestamped lines that appear after the first reflection are ignored
    (only this module writes to SELF.md, so they should not exist).
    """
    lines = text.splitlines()
    header_lines: list[str] = []
    reflections: list[str] = []
    header_done = False

    for line in lines:
        if not header_done and _DATE_RE.match(line):
            header_done = True
        if not header_done:
            header_lines.append(line)
        elif _DATE_RE.match(line):
            reflections.append(line)
        # non-dated body lines after header are silently dropped

    header = "\n".join(header_lines)
    return header, reflections


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def append_reflection(line: str) -> bool:
    """Append a datestamped first-person reflection line to SELF.md.

    Returns True on success, False if the line is rejected or an I/O error
    occurs on the existing file (to avoid overwriting an unreadable file).
    FIFO cap: the oldest entry is evicted when a 11th entry would be written.
    """
    line = (line or "").strip()
    if not _validate_reflection_authoring(line):
        return False

    path = _self_path()
    header = ""
    reflections: list[str] = []

    if path.exists():
        try:
            text = path.read_text(encoding="utf-8")
            header, reflections = _parse(text)
        except Exception as exc:
            _log.debug("self_journal: could not read %s: %s", path, exc)
            return False

    from hermes_time import now as _hermes_now
    date_str = _hermes_now().strftime("%Y-%m-%d")
    entry = f"{date_str} {line}"

    reflections.append(entry)
    reflections = reflections[-_MAX_ENTRIES:]

    if header.strip():
        content = header.rstrip("\n") + "\n\n" + "\n".join(reflections) + "\n"
    else:
        content = "\n".join(reflections) + "\n"

    try:
        _atomic_write(path, content)
    except Exception as exc:
        _log.debug("self_journal: atomic write failed: %s", exc)
        return False

    return True


def read_self_tail(max_lines: int = 10) -> str:
    """Return the last <=max_lines reflection entries from SELF.md.

    Returns '' if SELF.md is missing, empty, or contains only header/comment
    prose with no reflection entries.  max_lines is clamped to [0, 10].
    """
    n = max(0, min(int(max_lines), _MAX_ENTRIES))
    if n == 0:
        return ""

    path = _self_path()
    if not path.exists():
        return ""

    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""

    _, reflections = _parse(text)
    if not reflections:
        return ""

    return "\n".join(reflections[-n:])
