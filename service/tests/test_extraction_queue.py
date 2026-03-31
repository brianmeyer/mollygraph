from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

_HERE = Path(__file__).parent
_SERVICE_ROOT = _HERE.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from extraction.queue import ExtractionQueue


def _insert_job(
    queue: ExtractionQueue,
    *,
    job_id: str,
    status: str,
    created_at: datetime | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    retry_count: int = 0,
) -> None:
    now = datetime.now(UTC)
    created = (created_at or now).isoformat()
    reference_time = now.isoformat()

    with queue._get_conn() as conn:  # noqa: SLF001 - focused queue regression tests
        conn.execute(
            """
            INSERT INTO jobs (
                id, content, source, speaker, priority, reference_time,
                episode_id, status, created_at, started_at, completed_at,
                error, result_json, retry_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                f"{job_id} content",
                "session",
                None,
                1,
                reference_time,
                None,
                status,
                created,
                started_at.isoformat() if started_at else None,
                completed_at.isoformat() if completed_at else None,
                None,
                None,
                retry_count,
            ),
        )
        conn.commit()


def test_get_stuck_count_parses_iso_started_at(tmp_path) -> None:
    queue = ExtractionQueue(tmp_path / "queue.db")
    stale_started_at = datetime.now(UTC) - timedelta(minutes=10)

    _insert_job(
        queue,
        job_id="stale-job",
        status="processing",
        started_at=stale_started_at,
    )

    assert queue.get_stuck_count() == 1


def test_recover_stale_jobs_requeues_iso_processing_rows(tmp_path) -> None:
    queue = ExtractionQueue(tmp_path / "queue.db")
    stale_started_at = datetime.now(UTC) - timedelta(minutes=10)

    _insert_job(
        queue,
        job_id="stale-job",
        status="processing",
        started_at=stale_started_at,
    )

    assert queue.recover_stale_jobs() == 1

    with queue._get_conn() as conn:  # noqa: SLF001 - focused queue regression tests
        status, retry_count = conn.execute(
            "SELECT status, retry_count FROM jobs WHERE id = ?",
            ("stale-job",),
        ).fetchone()

    assert status == "pending"
    assert retry_count == 1


def test_claim_next_recovers_stale_jobs_before_selecting_pending(tmp_path) -> None:
    queue = ExtractionQueue(tmp_path / "queue.db")
    now = datetime.now(UTC)

    _insert_job(
        queue,
        job_id="older-stale",
        status="processing",
        created_at=now - timedelta(minutes=20),
        started_at=now - timedelta(minutes=10),
    )
    _insert_job(
        queue,
        job_id="newer-pending",
        status="pending",
        created_at=now - timedelta(minutes=5),
    )

    claimed = queue.claim_next()

    assert claimed is not None
    assert claimed.id == "older-stale"

    with queue._get_conn() as conn:  # noqa: SLF001 - focused queue regression tests
        stale_status, stale_retry = conn.execute(
            "SELECT status, retry_count FROM jobs WHERE id = ?",
            ("older-stale",),
        ).fetchone()
        pending_status = conn.execute(
            "SELECT status FROM jobs WHERE id = ?",
            ("newer-pending",),
        ).fetchone()[0]

    assert stale_status == "processing"
    assert stale_retry == 1
    assert pending_status == "pending"


def test_recover_stale_jobs_dead_letters_after_retry_limit(tmp_path) -> None:
    queue = ExtractionQueue(tmp_path / "queue.db")

    _insert_job(
        queue,
        job_id="almost-dead",
        status="processing",
        started_at=datetime.now(UTC) - timedelta(minutes=10),
        retry_count=queue._MAX_RETRIES - 1,  # noqa: SLF001 - regression target
    )

    assert queue.recover_stale_jobs() == 1

    with queue._get_conn() as conn:  # noqa: SLF001 - focused queue regression tests
        status, retry_count = conn.execute(
            "SELECT status, retry_count FROM jobs WHERE id = ?",
            ("almost-dead",),
        ).fetchone()

    assert status == "dead"
    assert retry_count == queue._MAX_RETRIES


def test_cleanup_old_parses_iso_completed_at(tmp_path) -> None:
    queue = ExtractionQueue(tmp_path / "queue.db")
    now = datetime.now(UTC)

    _insert_job(
        queue,
        job_id="old-completed",
        status="completed",
        completed_at=now - timedelta(days=10),
    )
    _insert_job(
        queue,
        job_id="recent-completed",
        status="completed",
        completed_at=now - timedelta(days=1),
    )

    queue.cleanup_old(days=7)

    with queue._get_conn() as conn:  # noqa: SLF001 - focused queue regression tests
        remaining_ids = [
            row[0]
            for row in conn.execute("SELECT id FROM jobs ORDER BY id").fetchall()
        ]

    assert remaining_ids == ["recent-completed"]
