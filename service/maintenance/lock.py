"""File-based maintenance lock for coordinating extraction ↔ maintenance.

Usage:
    - auditor.py acquires the lock during run_maintenance_cycle()
    - queue.py checks the lock before claiming jobs (skips if locked)

Uses fcntl.flock() for atomic locking with a 30-minute timeout guard.
Stale locks (dead PIDs) are cleaned up automatically.
"""
from __future__ import annotations

import fcntl
import logging
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import config

log = logging.getLogger(__name__)

_LOCK_PATH = config.GRAPH_MEMORY_DIR / "maintenance.lock"
_MAX_LOCK_AGE_SECONDS = 30 * 60  # 30 minutes


def _pid_alive(pid: int) -> bool:
    """Check if a process with given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _cleanup_stale_lock() -> bool:
    """Remove lock file if the owning PID is dead or lock is too old.
    
    Returns True if stale lock was cleaned up.
    """
    if not _LOCK_PATH.exists():
        return False

    try:
        content = _LOCK_PATH.read_text().strip()
        parts = content.split(":", 1)
        pid = int(parts[0])
        ts = float(parts[1]) if len(parts) > 1 else 0
    except (ValueError, OSError):
        # Corrupt lock file — remove it
        try:
            _LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return True

    # Check if PID is alive
    if not _pid_alive(pid):
        log.warning("Cleaning up stale maintenance lock (pid %d is dead)", pid)
        try:
            _LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return True

    # Check if lock is too old (timeout guard)
    if ts > 0 and (time.time() - ts) > _MAX_LOCK_AGE_SECONDS:
        log.warning(
            "Cleaning up expired maintenance lock (pid %d, age %.0fs > %ds)",
            pid, time.time() - ts, _MAX_LOCK_AGE_SECONDS,
        )
        try:
            _LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return True

    return False


def is_maintenance_locked() -> bool:
    """Check if maintenance is currently running.
    
    Cleans up stale locks automatically. Returns True if a valid lock
    is held by a live process.
    """
    _cleanup_stale_lock()
    return _LOCK_PATH.exists()


@contextmanager
def maintenance_lock():
    """Context manager: acquire maintenance lock for the duration of a cycle.
    
    Writes PID:timestamp to the lock file, uses fcntl.flock() for atomicity.
    Automatically releases on exit (including exceptions).
    
    Usage:
        with maintenance_lock():
            await run_maintenance_cycle()
    """
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_lock()

    fd = None
    try:
        fd = open(_LOCK_PATH, "w")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd.write(f"{os.getpid()}:{time.time()}")
        fd.flush()
        log.info("Maintenance lock acquired (pid=%d)", os.getpid())
        yield
    except BlockingIOError:
        log.warning("Could not acquire maintenance lock — another maintenance cycle is running")
        if fd:
            fd.close()
        raise RuntimeError("Maintenance lock already held")
    finally:
        if fd:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                fd.close()
            except OSError:
                pass
            try:
                _LOCK_PATH.unlink(missing_ok=True)
            except OSError:
                pass
            log.info("Maintenance lock released")
