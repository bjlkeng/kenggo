#!/usr/bin/env python3
"""Build the pinned CPU-only rules oracle; --offline reuses the verified archive."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import signal
import shutil
import subprocess
import tarfile
import time
import urllib.request


COMMIT = "fd0723fdbc0e9d82cf269c9630af8c27c57c07c4"
URL = f"https://codeload.github.com/lightvector/KataGo/tar.gz/{COMMIT}"
ARCHIVE_SHA256 = "5d663965f8c8886c1022b89565b9f1841045397bf7eb79c5cd57ea5881556a8a"
MAX_CAPTURE_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_BUILD_SECONDS = 540
SOURCE_PREFIXES = ("cpp/core/", "cpp/game/", "cpp/external/nlohmann_json/")
SOURCES = (
    "core/global.cpp", "core/hash.cpp", "core/rand.cpp", "core/rand_helpers.cpp",
    "core/sha2.cpp", "core/md5.cpp", "core/timer.cpp", "core/bsearch.cpp", "core/test.cpp",
    "game/board.cpp", "game/boardhistory.cpp", "game/rules.cpp",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_seconds(who: int) -> float:
    usage = resource.getrusage(who)
    return usage.ru_utime + usage.ru_stime


def fetch_archive(archive: Path, offline: bool, evidence: dict) -> None:
    if archive.exists():
        evidence["source_reused"] = True
    elif offline:
        raise ValueError(f"offline archive missing: {archive}")
    else:
        evidence["requests"] = 1
        temporary = archive.with_suffix(".download")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(URL, timeout=15) as response, temporary.open("wb") as target:
                evidence["http_status"] = response.status
                evidence["final_url"] = response.url
                count = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    count += len(chunk)
                    evidence["response_bytes"] = count
                    if count > MAX_CAPTURE_BYTES:
                        raise ValueError("source archive exceeds 64 MiB")
                    if time.monotonic() - started > 90:
                        raise ValueError("source download exceeds 90 seconds")
                    target.write(chunk)
            if sha256(temporary) != ARCHIVE_SHA256:
                raise ValueError("downloaded source archive SHA-256 does not match the pin")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    evidence["archive_bytes"] = archive.stat().st_size
    evidence["archive_sha256"] = sha256(archive)
    if archive.stat().st_size > MAX_CAPTURE_BYTES or evidence["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("cached source archive does not match the size/hash pin")


def extract_source(archive: Path, destination: Path, evidence: dict) -> None:
    entries = []
    total = 0
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if not path.parts or path.parts[0] != f"KataGo-{COMMIT}":
                raise ValueError("unexpected archive root")
            relative = PurePosixPath(*path.parts[1:])
            if not any(str(relative).startswith(prefix) for prefix in SOURCE_PREFIXES):
                continue
            if member.isdir():
                continue
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe source archive member")
            total += member.size
            if total > MAX_EXPANDED_BYTES:
                raise ValueError("expanded source exceeds 256 MiB")
            target = destination / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as data, target.open("wb") as output:
                shutil.copyfileobj(data, output)
            entries.append({"path": str(relative), "bytes": member.size, "sha256": sha256(target)})
    evidence["expanded_bytes"] = total
    evidence["source_files"] = entries


def build(args: argparse.Namespace, evidence: dict) -> None:
    root = Path(__file__).resolve().parents[1]
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "source.tar.gz"
    fetch_archive(archive, args.offline, evidence)
    source = cache / "source"
    extract_source(archive, source, evidence)
    compiler = shutil.which(args.cxx)
    if compiler is None:
        raise ValueError(f"C++ compiler not found: {args.cxx}")
    evidence["compiler"] = subprocess.check_output([compiler, "--version"], text=True).strip()
    harness = root / "tools" / "katago_reference.cpp"
    output = cache / "katago-reference"
    temporary_output = cache / "katago-reference.building"
    command = [compiler, "-std=c++17", "-O2", "-pthread", "-ffunction-sections", "-fdata-sections",
               "-Wl,--gc-sections", "-I", str(source / "cpp"), str(harness)]
    command += [str(source / "cpp" / path) for path in SOURCES]
    command += ["-o", str(temporary_output)]
    evidence["harness_sha256"] = sha256(harness)
    evidence["build_command"] = command
    evidence["compile_jobs"] = 1
    log_path = args.run_dir / "build.log"
    evidence["build_log"] = str(log_path)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return_code = process.wait(timeout=MAX_BUILD_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            temporary_output.unlink(missing_ok=True)
            raise
    evidence["build_exit_code"] = return_code
    if return_code:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(f"compiler failed; inspect {log_path}")
    temporary_output.replace(output)
    evidence["executable"] = str(output)
    evidence["executable_sha256"] = sha256(output)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="fail rather than fetch a missing archive")
    parser.add_argument("--cache-dir", type=Path, default=root / ".cache" / "katago-reference")
    parser.add_argument("--run-dir", type=Path, default=root / "runs" / "e1-rules-reference-2026-11-10")
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    own_cpu = cpu_seconds(resource.RUSAGE_SELF)
    child_cpu = cpu_seconds(resource.RUSAGE_CHILDREN)
    evidence = {"revision": COMMIT, "source_url": URL, "expected_archive_sha256": ARCHIVE_SHA256,
                "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "requests": 0, "response_bytes": 0, "source_reused": False, "gpu_hours": 0,
                "offline": args.offline, "ok": False}
    try:
        build(args, evidence)
        evidence["ok"] = True
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as error:
        evidence["error"] = str(error)
    finally:
        evidence["wall_seconds"] = time.monotonic() - started
        evidence["cpu_seconds"] = cpu_seconds(resource.RUSAGE_SELF) - own_cpu
        evidence["child_cpu_seconds"] = cpu_seconds(resource.RUSAGE_CHILDREN) - child_cpu
        evidence["max_child_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        evidence_path = args.run_dir / "build.json"
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
        summary = {key: value for key, value in evidence.items() if key != "source_files"}
        summary["evidence_path"] = str(evidence_path)
        print(json.dumps(summary))
    return 0 if evidence["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
