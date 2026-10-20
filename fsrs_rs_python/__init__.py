from __future__ import annotations
import importlib.machinery
import importlib.util
import itertools
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
_CRATE_DIR = Path(__file__).resolve().parent
_ROOT = _CRATE_DIR.parent
_MANIFEST_PATH = _CRATE_DIR / "Cargo.toml"
_MODULE_NAME = f"{__name__}.fsrs_rs_python"
_EXTENSION_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES) + (".abi3.pyd",)
_EXTENSION_BASENAMES = ("libfsrs_rs_python", "fsrs_rs_python")
_PYD_SUFFIX = next((suffix for suffix in _EXTENSION_SUFFIXES if suffix.endswith(".pyd")), None)
def _candidate_extensions() -> list[Path]:
    selected = os.environ.get("FSRS_EXTENSION_PATH")
    if selected:
        path = Path(selected).resolve()
        if not path.is_file():
            raise ImportError(f"FSRS_EXTENSION_PATH does not exist: {path}")
        return [path]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for profile, subdir, basename in itertools.product(
        ("release", "debug"), ("", "deps"), _EXTENSION_BASENAMES
    ):
        directory = _CRATE_DIR / "target" / profile / subdir
        if not directory.exists():
            continue
        for candidate in sorted(directory.glob(f"{basename}.*")):
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    if sys.platform == "win32" and _PYD_SUFFIX:
        for candidate in list(candidates):
            if candidate.suffix != ".dll":
                continue
            alias = candidate.with_name(candidate.stem + _PYD_SUFFIX)
            if not alias.exists():
                try:
                    shutil.copy2(candidate, alias)
                except PermissionError:
                    pass
            if alias not in seen:
                candidates.append(alias)
                seen.add(alias)
    return candidates
def _load_extension() -> ModuleType | None:
    for candidate in _candidate_extensions():
        if not any(candidate.name.endswith(suffix) for suffix in _EXTENSION_SUFFIXES):
            continue
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, candidate)
        if spec is None or spec.loader is None:
            continue
        try:
            module = importlib.util.module_from_spec(spec)
            sys.modules[_MODULE_NAME] = module
            spec.loader.exec_module(module)
            return module
        except Exception:
            sys.modules.pop(_MODULE_NAME, None)
    return None
def _build_extension() -> None:
    try:
        subprocess.run(
            ["cargo", "build", "--release", "--manifest-path", str(_MANIFEST_PATH),
             "--features", "pyo3/extension-module,pyo3/abi3-py312"],
            cwd=_ROOT,
            check=True,
        )
    except FileNotFoundError as exc:
        raise ImportError("Building fsrs-rs-python requires Rust/Cargo on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise ImportError("Failed to build the local fsrs-rs-python extension.") from exc
_ext = _load_extension()
if _ext is None:
    _build_extension()
    _ext = _load_extension()
    if _ext is None:
        raise ImportError("Built fsrs-rs-python, but the extension could not be imported.")
__doc__ = _ext.__doc__
__all__ = getattr(_ext, "__all__", [name for name in dir(_ext) if not name.startswith("_")])
globals().update({name: getattr(_ext, name) for name in __all__})
