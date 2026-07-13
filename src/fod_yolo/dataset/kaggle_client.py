"""Official Kaggle CLI download orchestration and safe ZIP extraction."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

from fod_yolo.dataset import DatasetDownloadError, KaggleAuthenticationError
from fod_yolo.hashing import atomic_replace_path, atomic_write_json, sha256_file
from fod_yolo.paths import ensure_within_root


@dataclass(frozen=True, slots=True)
class KaggleAuthentication:
    """Validated Kaggle authentication source without retaining secret values."""

    method: str
    config_file: Path | None = None


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """Immutable provenance for one downloaded source archive."""

    dataset_slug: str
    requested_version: str | None
    resolved_version: str | None
    downloaded_at_utc: str
    archive_path: str
    archive_size_bytes: int
    archive_sha256: str
    kaggle_cli_version: str

    def to_dict(self) -> dict[str, object]:
        """Return the stable source-manifest JSON schema."""

        return {
            "archive_path": self.archive_path,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "dataset_slug": self.dataset_slug,
            "downloaded_at_utc": self.downloaded_at_utc,
            "kaggle_cli_version": self.kaggle_cli_version,
            "requested_version": self.requested_version,
            "resolved_version": self.resolved_version,
            "schema_version": "1.0",
        }


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Paths and provenance returned by an idempotent download/extract operation."""

    archive_path: Path
    extracted_root: Path
    manifest_path: Path
    manifest: SourceManifest
    downloaded: bool
    extracted: bool


def resolve_kaggle_authentication(
    *,
    environment: Mapping[str, str] | None = None,
    home_directory: str | Path | None = None,
) -> KaggleAuthentication:
    """Require either a complete legacy environment pair or a local kaggle.json."""

    env = os.environ if environment is None else environment
    username = env.get("KAGGLE_USERNAME", "").strip()
    key = env.get("KAGGLE_KEY", "").strip()
    if username and key:
        return KaggleAuthentication(method="environment")
    if username or key:
        raise KaggleAuthenticationError(
            "KAGGLE_USERNAME and KAGGLE_KEY must both be set; one value is missing"
        )

    configured_directory = env.get("KAGGLE_CONFIG_DIR", "").strip()
    if configured_directory:
        config_directory = Path(configured_directory).expanduser().resolve()
    else:
        home = Path.home() if home_directory is None else Path(home_directory).expanduser()
        config_directory = (home / ".kaggle").resolve()
    config_file = config_directory / "kaggle.json"
    if config_file.is_file():
        return KaggleAuthentication(method="kaggle_json", config_file=config_file)

    raise KaggleAuthenticationError(
        "Kaggle credentials were not found. Set KAGGLE_USERNAME and KAGGLE_KEY, "
        f"or place kaggle.json at {config_file}"
    )


def build_kaggle_download_command(
    *,
    dataset_slug: str,
    output_directory: str | Path,
    version: str | int | None = None,
    force: bool = False,
    executable: str = "kaggle",
) -> tuple[str, ...]:
    """Build a shell-free official Kaggle CLI dataset download command."""

    slug = dataset_slug.strip().strip("/")
    if len(slug.split("/")) != 2 or any(not part for part in slug.split("/")):
        raise ValueError("Kaggle dataset slug must use owner/dataset format")
    dataset_reference = f"{slug}/{version}" if version is not None else slug
    command = [
        executable,
        "datasets",
        "download",
        "-d",
        dataset_reference,
        "-p",
        str(Path(output_directory).expanduser().resolve()),
    ]
    if force:
        command.append("--force")
    return tuple(command)


def download_and_extract_dataset(
    *,
    dataset_slug: str,
    raw_root: str | Path,
    archive_name: str,
    version: str | None,
    force_download: bool,
    force_extract: bool,
    environment: Mapping[str, str] | None = None,
) -> DownloadResult:
    """Idempotently download through Kaggle, hash, manifest, and safely extract."""

    resolved_raw_root = Path(raw_root).expanduser().resolve()
    downloads_root = resolved_raw_root / "downloads"
    archive_path = downloads_root / Path(archive_name).name
    extracted_root = resolved_raw_root / "extracted"
    manifest_path = resolved_raw_root / "source_manifest.json"
    downloads_root.mkdir(parents=True, exist_ok=True)

    existing_manifest = _read_source_manifest(manifest_path)
    archive_valid = (
        not force_download
        and archive_path.is_file()
        and existing_manifest is not None
        and existing_manifest.dataset_slug == dataset_slug
        and existing_manifest.requested_version == version
        and existing_manifest.archive_size_bytes == archive_path.stat().st_size
        and existing_manifest.archive_sha256 == sha256_file(archive_path)
    )

    downloaded = False
    if archive_valid:
        manifest = existing_manifest
    else:
        resolve_kaggle_authentication(environment=environment)
        executable = shutil.which("kaggle")
        if executable is None:
            raise DatasetDownloadError(
                "The official Kaggle CLI is not installed or the 'kaggle' command is not on PATH"
            )
        cli_version = _kaggle_cli_version(executable)
        _download_archive(
            executable=executable,
            dataset_slug=dataset_slug,
            version=version,
            downloads_root=downloads_root,
            archive_path=archive_path,
        )
        manifest = SourceManifest(
            dataset_slug=dataset_slug,
            requested_version=version,
            resolved_version=version,
            downloaded_at_utc=_utc_timestamp(),
            archive_path=str(Path("downloads") / archive_path.name),
            archive_size_bytes=archive_path.stat().st_size,
            archive_sha256=sha256_file(archive_path),
            kaggle_cli_version=cli_version,
        )
        atomic_write_json(manifest_path, manifest.to_dict())
        downloaded = True

    if manifest is None:  # Defensive guard for static analysis and corrupted state.
        raise DatasetDownloadError("A valid source manifest was not produced")

    extracted = safe_extract_zip(
        archive_path,
        extracted_root,
        force=force_extract or downloaded,
    )
    return DownloadResult(
        archive_path=archive_path,
        extracted_root=extracted_root,
        manifest_path=manifest_path,
        manifest=manifest,
        downloaded=downloaded,
        extracted=extracted,
    )


def safe_extract_zip(
    archive_path: str | Path,
    destination: str | Path,
    *,
    force: bool = False,
    maximum_uncompressed_bytes: int = 200 * 1024 * 1024 * 1024,
) -> bool:
    """Safely extract a ZIP into staging and atomically install the directory."""

    archive = Path(archive_path).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not archive.is_file():
        raise DatasetDownloadError(f"Dataset archive does not exist: {archive}")
    if target.exists() and not force:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))

    try:
        with zipfile.ZipFile(archive) as zip_file:
            members = zip_file.infolist()
            total_size = sum(member.file_size for member in members)
            if total_size > maximum_uncompressed_bytes:
                raise DatasetDownloadError(
                    f"Archive expands to {total_size} bytes, exceeding the configured safety limit"
                )
            for member in members:
                member_path = _validated_member_path(member, staging)
                if member.is_dir():
                    member_path.mkdir(parents=True, exist_ok=True)
                    continue
                member_path.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member) as source, member_path.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        _replace_directory(staging, target, force=force)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        _remove_directory(staging)
        raise DatasetDownloadError(f"Unable to safely extract {archive}: {exc}") from exc
    except DatasetDownloadError:
        _remove_directory(staging)
        raise
    return True


def _download_archive(
    *,
    executable: str,
    dataset_slug: str,
    version: str | None,
    downloads_root: Path,
    archive_path: Path,
) -> None:
    staging = Path(tempfile.mkdtemp(prefix=".kaggle-download.", suffix=".tmp", dir=downloads_root))
    command = build_kaggle_download_command(
        dataset_slug=dataset_slug,
        output_directory=staging,
        version=version,
        force=True,
        executable=executable,
    )
    try:
        result = subprocess.run(command, capture_output=True, check=False, text=True)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "Kaggle CLI download failed"
            raise DatasetDownloadError(detail)
        archives = sorted(staging.glob("*.zip"))
        if len(archives) != 1:
            raise DatasetDownloadError(
                f"Expected one ZIP archive from Kaggle, found {len(archives)} in {staging}"
            )
        atomic_replace_path(archives[0], archive_path)
    except OSError as exc:
        raise DatasetDownloadError(f"Unable to run Kaggle download: {exc}") from exc
    finally:
        _remove_directory(staging)


def _validated_member_path(member: zipfile.ZipInfo, staging_root: Path) -> Path:
    normalized_name = member.filename.replace("\\", "/")
    pure_path = PurePosixPath(normalized_name)
    unix_mode = member.external_attr >> 16
    if (
        pure_path.is_absolute()
        or ".." in pure_path.parts
        or any(":" in part for part in pure_path.parts)
        or stat.S_ISLNK(unix_mode)
    ):
        raise DatasetDownloadError(f"Unsafe archive member path: {member.filename!r}")
    candidate = staging_root.joinpath(*pure_path.parts)
    return ensure_within_root(candidate, staging_root, description="archive member")


def _replace_directory(staging: Path, target: Path, *, force: bool) -> None:
    if target.exists() and not force:
        _remove_directory(staging)
        return
    backup: Path | None = None
    if target.exists():
        backup = target.with_name(f".{target.name}.backup-{uuid4().hex}")
        atomic_replace_path(target, backup)
    try:
        atomic_replace_path(staging, target)
    except OSError:
        if backup is not None and backup.exists() and not target.exists():
            atomic_replace_path(backup, target)
        raise
    if backup is not None:
        _remove_directory(backup)


def _remove_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _read_source_manifest(path: Path) -> SourceManifest | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return SourceManifest(
            dataset_slug=str(value["dataset_slug"]),
            requested_version=(
                str(value["requested_version"])
                if value.get("requested_version") is not None
                else None
            ),
            resolved_version=(
                str(value["resolved_version"])
                if value.get("resolved_version") is not None
                else None
            ),
            downloaded_at_utc=str(value["downloaded_at_utc"]),
            archive_path=str(value["archive_path"]),
            archive_size_bytes=int(value["archive_size_bytes"]),
            archive_sha256=str(value["archive_sha256"]),
            kaggle_cli_version=str(value["kaggle_cli_version"]),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _kaggle_cli_version(executable: str) -> str:
    try:
        result = subprocess.run(
            (executable, "--version"),
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DatasetDownloadError(f"Unable to query Kaggle CLI version: {exc}") from exc
    if result.returncode != 0:
        raise DatasetDownloadError(result.stderr.strip() or "Kaggle CLI version query failed")
    return result.stdout.strip() or result.stderr.strip()


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
