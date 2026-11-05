"""Method-only extraction; study-specific commands and fixtures removed."""
from __future__ import annotations

import json

import importlib.metadata

import os

import platform

import tempfile

from pathlib import Path

from .contracts import canonical, digest, require

def file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def software_versions():
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "wfdb", "requests"):
        versions[package] = importlib.metadata.version(package)
    return versions

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation of final outputs is guarded by the case/run lock.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)

class Transaction:
    def __init__(self, directory, fingerprint, resume=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = self.directory / "active.lock"
        self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            manifest = self.directory / "binding.json"
            if manifest.exists():
                require(resume, "OUTPUT_EXISTS_USE_RESUME")
                require(json.loads(manifest.read_text("utf-8")) == fingerprint, "RESUME_HASH_MISMATCH")
            else:
                atomic_json(manifest, fingerprint)
            require(not (self.directory / "technical_failure.json").exists(), "TERMINAL_TECHNICAL_FAILURE_REVIEW_REQUIRED")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            # This removes only this runtime-owned empty lock, not experiment assets.
            self.lock.unlink()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            atomic_json(self.directory / "technical_failure.json", {"type": kind.__name__, "error": str(error)})
        self.close()

    def stage(self, name, inputs, operation):
        path = self.directory / (name + ".json")
        binding = digest(inputs)
        if path.exists():
            saved = json.loads(path.read_text("utf-8"))
            require(saved["input_sha256"] == binding and saved["output_sha256"] == digest(saved["output"]), "COMMITTED_STAGE_HASH_MISMATCH")
            return saved["output"]
        output = operation()
        atomic_json(path, {"input_sha256": binding, "output_sha256": digest(output), "output": output})
        return output
