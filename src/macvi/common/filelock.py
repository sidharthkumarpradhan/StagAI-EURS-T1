"""An exclusive lock file next to a shared file, for processes appending to it concurrently.

Two training lanes in one Kaggle session (``macvi.remote.lanes``) append to the same registry
and CSV logs. ``locked(path)`` holds ``<path>.lock`` (``fcntl.flock`` on Linux, ``msvcrt.locking``
on Windows) for the duration of the block; readers take it too, so nobody sees half a line.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def locked(path: str | Path, timeout_s: float = 600.0, optional: bool = False) -> Iterator[None]:
    """``optional``: when the lock file cannot be created (read-only folder), go ahead unlocked."""
    lock = Path(str(path) + ".lock")
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock, "a+b")
    except OSError:
        if not optional:
            raise
        yield
        return
    with handle as f:
        if os.name == "nt":
            import msvcrt

            def acquire() -> None:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

            def release() -> None:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire() -> None:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release() -> None:
                fcntl.flock(f, fcntl.LOCK_UN)

        end = time.monotonic() + timeout_s
        while True:  # non-blocking tries, so a stuck holder raises instead of hanging forever
            try:
                acquire()
                break
            except OSError:
                if time.monotonic() >= end:
                    raise TimeoutError(f"could not lock {lock} within {timeout_s:.0f} s") from None
                time.sleep(0.01)
        try:
            yield
        finally:
            release()
