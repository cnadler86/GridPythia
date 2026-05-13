"""Auto-updater via Git (GridPythia variant).

Same lifecycle as ZeroPythia's updater; uses structlog for logging.

Update lifecycle
----------------
1. Fetch remote info (read-only network call, run off the event loop).
2. Detect whether a newer version exists:
   - ``release`` mode: compare latest semver tag on remote vs. current HEAD tag.
   - ``master``  mode: compare remote master commit hash vs. local HEAD.
3. If an update is available:
   a. Pull / checkout the appropriate ref.
   b. Run ``uv sync --no-dev`` to update the venv.
   c. Send ``SIGTERM`` to own process so systemd restarts with new code.

Rate-limiting
-------------
One check per calendar day (UTC), tracked in memory.

Usage
-----
Create one ``AutoUpdater`` at startup; call
``await updater.check_and_update()`` after each successful plan publish.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
from datetime import date
from pathlib import Path
from typing import Optional

from structlog import get_logger

from GridPythia.config.server import UpdateMode

logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

_SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:[.\-].*)?$")


def _parse_semver(tag: str) -> Optional[tuple[int, int, int]]:
    """Return (major, minor, patch) or ``None`` if not a semver tag."""
    m = _SEMVER_RE.match(tag)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


# ── AutoUpdater ───────────────────────────────────────────────────────────────


class AutoUpdater:
    """Handles fetch, comparison, pull, dep-sync and graceful restart.

    Parameters
    ----------
    mode:
        Update policy (off / release / master).
    repo_path:
        Root of the git repository.  Defaults to the current working directory.
    branch:
        Remote branch to track in ``master`` mode.  Defaults to ``"master"``.
    remote:
        Name of the git remote.  Defaults to ``"origin"``.
    uv_executable:
        Path to the ``uv`` binary.  ``None`` → auto-detect from PATH.
    """

    def __init__(
        self,
        mode: UpdateMode,
        *,
        repo_path: Optional[Path] = None,
        branch: str = "master",
        remote: str = "origin",
        uv_executable: Optional[str] = None,
    ) -> None:
        self.mode = mode
        self._branch = branch
        self._remote = remote
        self._uv = uv_executable or "uv"
        self._repo_path = repo_path or Path.cwd()
        self._last_check_date: Optional[date] = None
        self._repo = None  # lazy-loaded gitpython Repo

    # ── Public API ────────────────────────────────────────────────────────────

    async def check_and_update(self) -> bool:
        """Check for an available update and apply it if found.

        Returns ``True`` when an update was applied and a restart requested.
        Safe to call repeatedly; performs at most one real check per day (UTC).
        """
        if self.mode is UpdateMode.OFF:
            return False

        today = date.today()
        if self._last_check_date == today:
            logger.debug("updater_already_checked_today", date=str(today))
            return False
        self._last_check_date = today

        try:
            repo = self._get_repo()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "updater_repo_open_failed", repo_path=str(self._repo_path), error=str(exc)
            )
            return False

        try:
            await asyncio.get_event_loop().run_in_executor(None, self._do_fetch, repo)
        except Exception as exc:  # noqa: BLE001
            logger.warning("updater_fetch_failed", error=str(exc))
            return False

        has_update, ref = self._detect_update(repo)
        if not has_update:
            logger.info("updater_no_update", mode=self.mode.value)
            return False

        logger.info("updater_applying", ref=ref, mode=self.mode.value)
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._apply_update, repo, ref)
        except Exception as exc:  # noqa: BLE001
            logger.error("updater_apply_failed", ref=ref, error=str(exc))
            return False

        dep_ok = await asyncio.get_event_loop().run_in_executor(None, self._sync_dependencies)
        if not dep_ok:
            logger.warning("updater_dep_sync_failed_restarting_anyway")

        logger.info("updater_restart_requested", pid=os.getpid())
        self._request_restart()
        return True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_repo(self):
        if self._repo is None:
            from git import Repo  # noqa: PLC0415

            self._repo = Repo(self._repo_path, search_parent_directories=True)
        return self._repo

    def _do_fetch(self, repo) -> None:
        repo.remotes[self._remote].fetch(tags=True)

    def _detect_update(self, repo) -> tuple[bool, str]:
        if self.mode is UpdateMode.RELEASE:
            return self._check_new_release(repo)
        if self.mode is UpdateMode.MASTER:
            return self._check_master_update(repo)
        return False, ""

    def _check_new_release(self, repo) -> tuple[bool, str]:
        all_tags = [t for t in repo.tags if _parse_semver(t.name)]
        if not all_tags:
            logger.debug("updater_no_semver_tags")
            return False, ""

        latest_tag = max(all_tags, key=lambda t: _parse_semver(t.name) or (0, 0, 0))

        try:
            current_tag_name = next(t.name for t in repo.tags if t.commit == repo.head.commit)
        except StopIteration:
            current_tag_name = None

        if current_tag_name is None:
            logger.info("updater_head_not_on_tag", latest=latest_tag.name)
            return True, latest_tag.name

        current_ver = _parse_semver(current_tag_name)
        latest_ver = _parse_semver(latest_tag.name)

        if latest_ver is not None and current_ver is not None and latest_ver > current_ver:
            logger.info("updater_new_release", latest=latest_tag.name, current=current_tag_name)
            return True, latest_tag.name

        logger.debug("updater_already_latest_release", tag=current_tag_name)
        return False, ""

    def _check_master_update(self, repo) -> tuple[bool, str]:
        try:
            remote_ref = repo.remotes[self._remote].refs[self._branch]
        except (IndexError, AttributeError):
            logger.warning(
                "updater_remote_ref_not_found",
                remote=self._remote,
                branch=self._branch,
            )
            return False, ""

        remote_sha = remote_ref.commit.hexsha
        local_sha = repo.head.commit.hexsha

        if remote_sha != local_sha:
            logger.info(
                "updater_master_ahead",
                remote=remote_sha[:8],
                local=local_sha[:8],
            )
            return True, f"{self._remote}/{self._branch}"

        logger.debug("updater_master_up_to_date", branch=self._branch)
        return False, ""

    def _apply_update(self, repo, ref: str) -> None:
        if self.mode is UpdateMode.RELEASE:
            logger.info("updater_checkout_tag", ref=ref)
            repo.git.checkout(ref)
        elif self.mode is UpdateMode.MASTER:
            logger.info("updater_pull_branch", remote=self._remote, branch=self._branch)
            repo.remotes[self._remote].pull(self._branch)

    def _sync_dependencies(self) -> bool:
        try:
            result = subprocess.run(  # noqa: S603
                [self._uv, "sync", "--no-dev"],
                cwd=self._repo_path,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode == 0:
                logger.info("updater_dep_sync_ok")
                return True
            logger.error(
                "updater_dep_sync_failed",
                returncode=result.returncode,
                stderr=result.stderr.strip(),
            )
            return False
        except FileNotFoundError:
            logger.warning("updater_uv_not_found", uv=self._uv)
            return False
        except subprocess.TimeoutExpired:
            logger.error("updater_dep_sync_timeout")
            return False

    @staticmethod
    def _request_restart() -> None:
        """Send SIGTERM to own process so systemd triggers a restart."""
        os.kill(os.getpid(), signal.SIGTERM)
