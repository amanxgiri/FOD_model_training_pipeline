"""Environment inspection and explicit PyTorch installation support."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from fod_yolo.hashing import atomic_write_text

ENVIRONMENT_EXIT_CODE = 6
_TORCH_PROFILE_PATTERN = re.compile(r"(?:cpu|cu\d{3}|rocm\d+\.\d+|xpu)", re.IGNORECASE)
_PACKAGE_VERSION_PATTERN = re.compile(r"[0-9][A-Za-z0-9.+_-]*")


class EnvironmentValidationError(RuntimeError):
    """Raised when required packages, a model check, or CUDA validation fails."""


class TorchInstallationError(RuntimeError):
    """Raised when an explicit PyTorch installation command fails."""


@dataclass(frozen=True, slots=True)
class PackageReport:
    """Import and distribution metadata for one required package."""

    module: str
    available: bool
    version: str | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "available": self.available,
            "error": self.error,
            "module": self.module,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class GpuReport:
    """GPU information reported through PyTorch."""

    index: int
    name: str
    total_memory_bytes: int
    compute_capability: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "compute_capability": self.compute_capability,
            "index": self.index,
            "name": self.name,
            "total_memory_bytes": self.total_memory_bytes,
        }


@dataclass(frozen=True, slots=True)
class NvidiaSmiGpuReport:
    """One row from a stable nvidia-smi CSV query."""

    index: int
    name: str
    memory_total_mb: int
    driver_version: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "driver_version": self.driver_version,
            "index": self.index,
            "memory_total_mb": self.memory_total_mb,
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class NvidiaSmiReport:
    """Availability and parsed output from the nvidia-smi command."""

    available: bool
    executable: str | None
    return_code: int | None
    gpus: tuple[NvidiaSmiGpuReport, ...]
    error: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "available": self.available,
            "error": self.error,
            "executable": self.executable,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "return_code": self.return_code,
        }


@dataclass(frozen=True, slots=True)
class GitReport:
    """Current source revision and dirty-state metadata."""

    commit: str | None
    dirty: bool | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {"commit": self.commit, "dirty": self.dirty, "error": self.error}


@dataclass(frozen=True, slots=True)
class ModelCheckReport:
    """Result of resolving an Ultralytics model checkpoint."""

    checkpoint: str | None
    attempted: bool
    passed: bool | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "attempted": self.attempted,
            "checkpoint": self.checkpoint,
            "error": self.error,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """Reproducibility and accelerator metadata for one machine inspection."""

    generated_at_utc: str
    python_version: str
    python_executable: str
    python_implementation: str
    operating_system: str
    platform_release: str
    machine: str
    processor: str
    packages: dict[str, PackageReport]
    cuda_available: bool
    cuda_device_count: int
    cuda_current_device: int | None
    torch_cuda_version: str | None
    cudnn_version: int | None
    cuda_smoke_test: str
    cuda_error: str | None
    gpus: tuple[GpuReport, ...]
    nvidia_smi: NvidiaSmiReport
    git: GitReport
    model_check: ModelCheckReport

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON schema used by environment reports."""

        return {
            "cuda": {
                "available": self.cuda_available,
                "current_device": self.cuda_current_device,
                "cudnn_version": self.cudnn_version,
                "device_count": self.cuda_device_count,
                "error": self.cuda_error,
                "smoke_test": self.cuda_smoke_test,
                "torch_cuda_version": self.torch_cuda_version,
            },
            "generated_at_utc": self.generated_at_utc,
            "git": self.git.to_dict(),
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "model_check": self.model_check.to_dict(),
            "nvidia_smi": self.nvidia_smi.to_dict(),
            "packages": {
                name: package.to_dict() for name, package in sorted(self.packages.items())
            },
            "platform": {
                "machine": self.machine,
                "operating_system": self.operating_system,
                "processor": self.processor,
                "release": self.platform_release,
            },
            "python": {
                "executable": self.python_executable,
                "implementation": self.python_implementation,
                "version": self.python_version,
            },
            "schema_version": "1.0",
        }


_PACKAGE_SPECS = {
    "opencv": ("cv2", ("opencv-python", "opencv-python-headless")),
    "torch": ("torch", ("torch",)),
    "torchvision": ("torchvision", ("torchvision",)),
    "ultralytics": ("ultralytics", ("ultralytics",)),
}


def collect_environment_report(
    *,
    project_root: str | Path | None = None,
    run_cuda_test: bool = False,
    model_checkpoint: str | None = None,
) -> EnvironmentReport:
    """Inspect packages, platform, Git, NVIDIA tools, CUDA, and an optional model."""

    packages = {
        name: inspect_package(module_name, distributions)
        for name, (module_name, distributions) in _PACKAGE_SPECS.items()
    }
    (
        cuda_available,
        cuda_device_count,
        cuda_current_device,
        torch_cuda_version,
        cudnn_version,
        cuda_smoke_test,
        cuda_error,
        gpus,
    ) = _inspect_torch_cuda(packages["torch"], run_cuda_test=run_cuda_test)

    root = Path.cwd() if project_root is None else Path(project_root).expanduser().resolve()
    return EnvironmentReport(
        generated_at_utc=_utc_timestamp(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        python_implementation=platform.python_implementation(),
        operating_system=platform.system(),
        platform_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        packages=packages,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_current_device=cuda_current_device,
        torch_cuda_version=torch_cuda_version,
        cudnn_version=cudnn_version,
        cuda_smoke_test=cuda_smoke_test,
        cuda_error=cuda_error,
        gpus=gpus,
        nvidia_smi=inspect_nvidia_smi(),
        git=inspect_git(root),
        model_check=inspect_model_checkpoint(model_checkpoint, packages["ultralytics"]),
    )


def inspect_package(module_name: str, distributions: tuple[str, ...]) -> PackageReport:
    """Import a package and resolve its installed distribution version."""

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return PackageReport(
            module=module_name,
            available=False,
            version=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    version: str | None = None
    for distribution in distributions:
        try:
            version = importlib.metadata.version(distribution)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    if version is None:
        module_version = getattr(module, "__version__", None)
        version = str(module_version) if module_version is not None else None

    return PackageReport(module=module_name, available=True, version=version, error=None)


def inspect_nvidia_smi(*, timeout_seconds: float = 10.0) -> NvidiaSmiReport:
    """Run a fixed nvidia-smi query and parse its stable CSV fields."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return NvidiaSmiReport(
            available=False,
            executable=None,
            return_code=None,
            gpus=(),
            error="nvidia-smi was not found on PATH",
        )

    command = (
        executable,
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return NvidiaSmiReport(
            available=True,
            executable=executable,
            return_code=None,
            gpus=(),
            error=f"{type(exc).__name__}: {exc}",
        )

    if result.returncode != 0:
        error = result.stderr.strip() or "nvidia-smi returned a non-zero exit code"
        return NvidiaSmiReport(
            available=True,
            executable=executable,
            return_code=result.returncode,
            gpus=(),
            error=error,
        )

    try:
        gpus = parse_nvidia_smi_output(result.stdout)
    except ValueError as exc:
        return NvidiaSmiReport(
            available=True,
            executable=executable,
            return_code=result.returncode,
            gpus=(),
            error=str(exc),
        )
    return NvidiaSmiReport(
        available=True,
        executable=executable,
        return_code=result.returncode,
        gpus=gpus,
        error=None,
    )


def parse_nvidia_smi_output(output: str) -> tuple[NvidiaSmiGpuReport, ...]:
    """Parse index, name, memory MiB, and driver version CSV rows."""

    rows: list[NvidiaSmiGpuReport] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"Malformed nvidia-smi CSV row {line_number}: {line!r}")
        try:
            index = int(fields[0])
            memory_total_mb = int(fields[2])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric nvidia-smi row {line_number}: {line!r}") from exc
        rows.append(
            NvidiaSmiGpuReport(
                index=index,
                name=fields[1],
                memory_total_mb=memory_total_mb,
                driver_version=fields[3],
            )
        )
    return tuple(rows)


def inspect_git(project_root: str | Path) -> GitReport:
    """Return the current Git commit and dirty state without modifying the repository."""

    root = Path(project_root).expanduser().resolve()
    try:
        commit_result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        status_result = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GitReport(commit=None, dirty=None, error=f"{type(exc).__name__}: {exc}")

    if commit_result.returncode != 0 or status_result.returncode != 0:
        error = commit_result.stderr.strip() or status_result.stderr.strip() or "Git query failed"
        return GitReport(commit=None, dirty=None, error=error)
    return GitReport(
        commit=commit_result.stdout.strip(),
        dirty=bool(status_result.stdout.strip()),
        error=None,
    )


def inspect_model_checkpoint(
    checkpoint: str | None,
    ultralytics_package: PackageReport,
) -> ModelCheckReport:
    """Instantiate an Ultralytics YOLO checkpoint only when explicitly requested."""

    if checkpoint is None:
        return ModelCheckReport(checkpoint=None, attempted=False, passed=None, error=None)
    if not ultralytics_package.available:
        return ModelCheckReport(
            checkpoint=checkpoint,
            attempted=True,
            passed=False,
            error="Ultralytics is not importable",
        )

    try:
        ultralytics = importlib.import_module("ultralytics")
        ultralytics.YOLO(checkpoint)
    except Exception as exc:
        return ModelCheckReport(
            checkpoint=checkpoint,
            attempted=True,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return ModelCheckReport(checkpoint=checkpoint, attempted=True, passed=True, error=None)


def validate_environment(
    report: EnvironmentReport,
    *,
    require_cuda: bool,
    require_model_check: bool,
) -> None:
    """Raise one actionable error when the inspected environment is incomplete."""

    missing_packages = [name for name, package in report.packages.items() if not package.available]
    if missing_packages:
        raise EnvironmentValidationError(
            f"Required packages are unavailable: {', '.join(sorted(missing_packages))}"
        )
    if require_cuda and not report.cuda_available:
        raise EnvironmentValidationError("CUDA is required but torch.cuda.is_available() is false")
    if require_cuda and report.cuda_smoke_test != "passed":
        detail = f": {report.cuda_error}" if report.cuda_error else ""
        raise EnvironmentValidationError(f"CUDA tensor smoke test did not pass{detail}")
    if require_model_check and report.model_check.passed is not True:
        detail = f": {report.model_check.error}" if report.model_check.error else ""
        raise EnvironmentValidationError(f"YOLO checkpoint validation did not pass{detail}")


def format_environment_summary(report: EnvironmentReport) -> str:
    """Create a concise human-readable summary for console output."""

    package_versions = ", ".join(
        f"{name}={package.version or 'unavailable'}"
        for name, package in sorted(report.packages.items())
    )
    gpu_names = ", ".join(gpu.name for gpu in report.gpus) or "none"
    return "\n".join(
        (
            f"Python: {report.python_version} ({report.python_executable})",
            f"Platform: {report.operating_system} {report.platform_release} {report.machine}",
            f"Packages: {package_versions}",
            f"CUDA: available={report.cuda_available} devices={report.cuda_device_count} ",
            f"GPUs: {gpu_names}",
            f"CUDA smoke test: {report.cuda_smoke_test}",
        )
    )


def resolve_torch_index_url(*, index_url: str | None, profile: str | None) -> str:
    """Resolve a named compute profile or validate an official PyTorch wheel URL."""

    if (index_url is None) == (profile is None):
        raise ValueError("Provide exactly one of index_url or profile")

    if profile is not None:
        normalized_profile = profile.strip().lower()
        if _TORCH_PROFILE_PATTERN.fullmatch(normalized_profile) is None:
            raise ValueError("Invalid PyTorch profile; expected cpu, cuNNN, rocmN.N, or xpu")
        return f"https://download.pytorch.org/whl/{normalized_profile}"

    assert index_url is not None
    normalized_url = index_url.strip().rstrip("/")
    parsed = urlparse(normalized_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "download.pytorch.org"
        or not parsed.path.startswith("/whl/")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "The official PyTorch index URL must be HTTPS under https://download.pytorch.org/whl/"
        )
    return normalized_url


def build_torch_install_command(
    *,
    index_url: str,
    torch_version: str | None = None,
    torchvision_version: str | None = None,
    python_executable: str | Path | None = None,
) -> tuple[str, ...]:
    """Build a shell-free pip command for the selected interpreter and official index."""

    validated_index = resolve_torch_index_url(index_url=index_url, profile=None)
    torch_spec = _package_spec("torch", torch_version)
    torchvision_spec = _package_spec("torchvision", torchvision_version)
    executable = str(python_executable or sys.executable)
    return (
        executable,
        "-m",
        "pip",
        "install",
        torch_spec,
        torchvision_spec,
        "--index-url",
        validated_index,
    )


def install_torch_packages(command: tuple[str, ...]) -> dict[str, object]:
    """Execute a prebuilt pip command and return installed version metadata."""

    try:
        result = subprocess.run(command, capture_output=True, check=False, text=True)
    except OSError as exc:
        raise TorchInstallationError(f"Unable to execute pip: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "pip install failed"
        raise TorchInstallationError(detail)

    packages = {
        name: inspect_package(module_name, distributions).to_dict()
        for name, (module_name, distributions) in _PACKAGE_SPECS.items()
        if name in {"torch", "torchvision"}
    }
    return {
        "command": list(command),
        "completed_at_utc": _utc_timestamp(),
        "packages": packages,
        "python_executable": command[0],
        "status": "success",
    }


def capture_pip_freeze(*, python_executable: str | Path | None = None) -> tuple[str, ...]:
    """Capture a sorted package freeze for the selected Python interpreter."""

    executable = str(python_executable or sys.executable)
    try:
        result = subprocess.run(
            (executable, "-m", "pip", "freeze"),
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EnvironmentValidationError(f"Unable to capture pip freeze: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "pip freeze failed"
        raise EnvironmentValidationError(detail)
    return tuple(sorted(line for line in result.stdout.splitlines() if line.strip()))


def write_environment_freeze(
    path: str | Path,
    *,
    python_executable: str | Path | None = None,
) -> Path:
    """Capture and atomically write a reproducibility package freeze."""

    lines = capture_pip_freeze(python_executable=python_executable)
    content = "\n".join(lines)
    return atomic_write_text(path, f"{content}\n" if content else "")


def _inspect_torch_cuda(
    torch_package: PackageReport,
    *,
    run_cuda_test: bool,
) -> tuple[
    bool,
    int,
    int | None,
    str | None,
    int | None,
    str,
    str | None,
    tuple[GpuReport, ...],
]:
    if not torch_package.available:
        smoke_status = "failed" if run_cuda_test else "not_run"
        error = "PyTorch is not importable" if run_cuda_test else None
        return False, 0, None, None, None, smoke_status, error, ()

    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        current_device = int(torch.cuda.current_device()) if device_count else None
        torch_cuda_version_raw = getattr(torch.version, "cuda", None)
        torch_cuda_version = (
            str(torch_cuda_version_raw) if torch_cuda_version_raw is not None else None
        )
        cudnn_version_raw = torch.backends.cudnn.version()
        cudnn_version = int(cudnn_version_raw) if cudnn_version_raw is not None else None
        gpus = tuple(_torch_gpu_report(torch, index) for index in range(device_count))
    except Exception as exc:
        return False, 0, None, None, None, "failed", f"{type(exc).__name__}: {exc}", ()

    smoke_status = "not_run"
    smoke_error: str | None = None
    if run_cuda_test:
        if not cuda_available:
            smoke_status = "failed"
            smoke_error = "CUDA is unavailable"
        else:
            try:
                tensor = torch.tensor([1.0], device="cuda")
                result = float((tensor * 2).item())
                torch.cuda.synchronize()
                if result != 2.0:
                    raise RuntimeError(f"Unexpected CUDA tensor result: {result}")
                smoke_status = "passed"
            except Exception as exc:
                smoke_status = "failed"
                smoke_error = f"{type(exc).__name__}: {exc}"

    return (
        cuda_available,
        device_count,
        current_device,
        torch_cuda_version,
        cudnn_version,
        smoke_status,
        smoke_error,
        gpus,
    )


def _torch_gpu_report(torch: object, index: int) -> GpuReport:
    properties = torch.cuda.get_device_properties(index)  # type: ignore[attr-defined]
    capability: str | None = None
    major = getattr(properties, "major", None)
    minor = getattr(properties, "minor", None)
    if major is not None and minor is not None:
        capability = f"{major}.{minor}"
    return GpuReport(
        index=index,
        name=str(properties.name),
        total_memory_bytes=int(properties.total_memory),
        compute_capability=capability,
    )


def _package_spec(name: str, version: str | None) -> str:
    if version is None:
        return name
    normalized = version.strip()
    if _PACKAGE_VERSION_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"Invalid {name} version: {version!r}")
    return f"{name}=={normalized}"


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
