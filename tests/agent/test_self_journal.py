"""Tests for agent/self_journal.py — SELF.md journal infrastructure."""

from pathlib import Path

from agent.self_journal import append_reflection, read_self_tail


def _self_path(tmp_path_hermes_home):
    """Resolve the SELF.md path inside the per-test HERMES_HOME."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "memories" / "SELF.md"


# ── Writer tests ────────────────────────────────────────────────────────────

class TestAppendReflection:
    def test_datestamped_first_person(self):
        result = append_reflection("Learned to keep responses concise.")
        assert result is True
        from hermes_constants import get_hermes_home
        content = (get_hermes_home() / "memories" / "SELF.md").read_text()
        import re
        assert re.search(r"^\d{4}-\d{2}-\d{2} Learned to keep responses concise\.", content, re.MULTILINE)

    def test_fifo_cap_10(self):
        lines = [f"Reflection number {i}." for i in range(12)]
        for ln in lines:
            append_reflection(ln)

        tail = read_self_tail()
        tail_lines = tail.splitlines()
        assert len(tail_lines) == 10
        # oldest 2 evicted
        assert "Reflection number 0." not in tail
        assert "Reflection number 1." not in tail
        # newest present
        assert "Reflection number 11." in tail

    def test_header_preserved_atomic_shape(self):
        from hermes_constants import get_hermes_home
        self_path = get_hermes_home() / "memories" / "SELF.md"
        header_text = "# SELF journal\n_First-person._\n"
        self_path.write_text(header_text, encoding="utf-8")

        append_reflection("I now prefer brevity.")

        content = self_path.read_text(encoding="utf-8")
        # Header preserved at top
        assert content.startswith("# SELF journal\n_First-person._\n")
        # Reflection entry present
        import re
        assert re.search(r"\d{4}-\d{2}-\d{2} I now prefer brevity\.", content)

    def test_guard_rejects_second_person(self):
        result = append_reflection("You are helpful")
        assert result is False
        assert read_self_tail() == ""

    def test_guard_rejects_second_person_variant(self):
        result = append_reflection("You're always right.")
        assert result is False
        assert read_self_tail() == ""

    def test_guard_rejects_empty_string(self):
        assert append_reflection("") is False
        assert read_self_tail() == ""

    def test_guard_rejects_whitespace_only(self):
        assert append_reflection("   ") is False
        assert read_self_tail() == ""

    def test_returns_false_without_writing_on_guard_fail(self):
        # Ensure no file is created on rejection
        from hermes_constants import get_hermes_home
        self_path = get_hermes_home() / "memories" / "SELF.md"
        assert not self_path.exists()
        append_reflection("You are not allowed")
        assert not self_path.exists()


# ── Reader tests ────────────────────────────────────────────────────────────

class TestReadSelfTail:
    def test_returns_empty_when_file_missing(self):
        assert read_self_tail() == ""

    def test_returns_empty_for_empty_file(self):
        from hermes_constants import get_hermes_home
        (get_hermes_home() / "memories" / "SELF.md").write_text("", encoding="utf-8")
        assert read_self_tail() == ""

    def test_returns_empty_for_header_only(self):
        from hermes_constants import get_hermes_home
        seed = "# SELF journal\n_First-person reflections only._\n"
        (get_hermes_home() / "memories" / "SELF.md").write_text(seed, encoding="utf-8")
        assert read_self_tail() == ""

    def test_max_and_clamp(self):
        for i in range(12):
            append_reflection(f"Entry {i}.")

        tail_default = read_self_tail()
        assert len(tail_default.splitlines()) == 10

        tail_3 = read_self_tail(3)
        assert len(tail_3.splitlines()) == 3

        tail_50 = read_self_tail(50)
        assert len(tail_50.splitlines()) <= 10

    def test_max_lines_zero_returns_empty(self):
        append_reflection("Something I learned today.")
        assert read_self_tail(0) == ""

    def test_returns_last_n_in_order(self):
        for i in range(5):
            append_reflection(f"Step {i} complete.")

        tail = read_self_tail(3)
        lines = tail.splitlines()
        assert len(lines) == 3
        assert "Step 2 complete." in lines[0]
        assert "Step 3 complete." in lines[1]
        assert "Step 4 complete." in lines[2]
