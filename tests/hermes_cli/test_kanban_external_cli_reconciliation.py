"""External holds survive the short-lived CLI and all reclaim sweeps."""
import os
from pathlib import Path
import subprocess
import sys

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
from hermes_cli import kanban_db_dispatch as dispatch


def test_external_cli_exit_survives_reconciliation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with connect() as conn:
        external = kb.create_task(conn, title="interactive audit")
        orphan = kb.create_task(conn, title="broken worker")
        kb.claim_task(conn, orphan)
        conn.execute("UPDATE tasks SET claim_expires=NULL WHERE id=?", (orphan,))
        conn.commit()
    cli = subprocess.run(
        [sys.executable, "-c", "from hermes_cli.main import main; main()",
         "kanban", "claim", external, "--external"],
        env=dict(os.environ), capture_output=True, text=True, timeout=60,
    )
    assert cli.returncode == 0, cli.stdout + cli.stderr
    # subprocess.run has waited for CLI termination: its PID is no longer live.
    with connect() as conn:
        before = kb.get_task(conn, external)
        assert before.claim_kind == "external"
        assert before.claim_expires is None
        result = dispatch.DispatchResult()
        dispatch._run_reclaim_phase(
            conn, result, stale_timeout_seconds=1,
            failure_limit=2, reconcile_orphans=True,
        )
        assert orphan in result.reconciled_orphans
        assert external not in result.reconciled_orphans
        after = kb.get_task(conn, external)
        assert after.status == "running"
        assert after.current_run_id == before.current_run_id
        assert after.worker_pid is None
        assert after.claim_kind == "external"
