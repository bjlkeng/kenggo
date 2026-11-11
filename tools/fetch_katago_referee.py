#!/usr/bin/env python3
"""Fetch the pinned CPU referee, or verify its cache without requests or writes."""

from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import selectors
import shutil
import signal
import stat
import struct
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile


PROFILE_SHA256 = "2ba2d542cd66073d1c7c34aae4598c776fb5adcf910439ed9fd885492a51a261"
MAX_WALL_SECONDS = 300
MAX_HTTP_REQUESTS = 8
MAX_DOWNLOAD_BYTES = 64 * 1024**2
MAX_ZIP_ENTRIES = 64
MAX_ZIP_EXPANDED_BYTES = 128 * 1024**2
MAX_EXTRACT_CPU_SECONDS = 60
MAX_EXTRACT_WALL_SECONDS = 90
MAX_STDERR_BYTES = 64 * 1024
APPIMAGE_OFFSET = 944632
OFFICIAL_HOSTS = {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com",
                  "media.katagotraining.org"}


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    size: int
    sha256: str


ARCHIVE = Artifact("katago-v1.18.1-eigenavx2-linux-x64.zip",
    "https://github.com/lightvector/KataGo/releases/download/v1.18.1/katago-v1.18.1-eigenavx2-linux-x64.zip",
    41821245, "33e79780dbe3bf6ee859e16f64952cdfc90f7210c8f71ad978ffcba85ad20d79")
APPIMAGE = Artifact("katago", ARCHIVE.url + "#katago", 42125816,
    "05c3450274bb8008e57e3011a7c80eb6e370ea395d29ab49c400bda9473766b1")
MODEL = Artifact("kata1-b10c128-s1141046784-d204142634.txt.gz",
    "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b10c128-s1141046784-d204142634.txt.gz",
    14466254, "3d8a24697ba25fe4da39af4c2b6bd405907b0ad8295322f5a550fa2d8fe4a2f4")
RUNTIME = (
    Artifact("usr/bin/katago", APPIMAGE.url, 142284992,
             "c0ed553a775fd9af585717ff00ddeb5007bf15d283caf148d237e0ab24f6f308"),
    Artifact("usr/lib/libbz2.so.1.0", APPIMAGE.url, 74848,
             "5e516f77fc36dd924fdf02c8489a217f55fa1548883d32c3a5e041fb25d47d6e"),
    Artifact("usr/lib/libz.so.1", APPIMAGE.url, 108936,
             "64c206f0146cc58bbddc4f22054436f4ff278f5a554aa3ce6921ddf7e9133370"),
    Artifact("usr/lib/libzip.so.4", APPIMAGE.url, 118480,
             "05746c16c79c34f3b408c5a6a5ddf9bd18e61c1cfd35f587e215ea37f23fe4ad"),
    Artifact("AppRun", APPIMAGE.url, 149,
             "2ddcaa8755c73b980e895d1466fa8589b998c47686846a995fe3a8e258b4c122"),
)
PROVENANCE = {
    "engine_release": "https://github.com/lightvector/KataGo/releases/tag/v1.18.1",
    "engine_revision": "92ee95c0a4b25fec214da00951ab69e97e207729",
    "engine_license": "MIT, with separate third-party notices",
    "engine_license_url": "https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/LICENSE",
    "model_catalog": "https://katagotraining.org/networks/",
    "model_license": "Publisher's MIT-style grant for network weights; separate from engine source license",
    "model_license_url": "https://katagotraining.org/network_license/",
    "bundled_library_license_urls": {"libbz2": "https://sourceware.org/bzip2/",
                                    "libz": "https://zlib.net/zlib_license.html",
                                    "libzip": "https://libzip.org/license/"},
    "native_verifier": "Build separately with tools/build_katago_reference.py; this tool does not fetch or build it.",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


class Deadline:
    def __init__(self, seconds=MAX_WALL_SECONDS):
        self.end = time.monotonic() + seconds

    def remaining(self):
        left = self.end - time.monotonic()
        if left <= 0:
            raise TimeoutError("Acquisition wall limit reached")
        return left


def safe_path(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Unsafe artifact path")
    target = Path(root) / str(relative)
    if target.is_symlink() or not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact path traverses a symlink outside its cache")
    return target


def verify_file(path, artifact):
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != artifact.size
            or sha256(path) != artifact.sha256):
        raise ValueError(f"Artifact size/hash mismatch: {artifact.name}")
    return {"name": artifact.name, "url": artifact.url, "bytes": artifact.size, "sha256": artifact.sha256}


def validate_profile(path, expected_sha256):
    path = Path(path)
    if path.stat().st_size > 128_000 or sha256(path) != expected_sha256:
        raise ValueError("Referee profile differs from its external pin")
    profile = json.loads(path.read_text())
    if not isinstance(profile, dict) or not isinstance(profile.get("artifacts"), dict):
        raise ValueError("Invalid referee profile object")
    pins = profile.get("artifacts", {})
    libraries = {item.name: item.sha256 for item in RUNTIME if item.name.startswith("usr/lib/")}
    if (profile.get("schema") not in ("katago_referee_profile_v1", "katago_referee_profile_v2")
            or pins.get("engine") != RUNTIME[0].sha256 or pins.get("model") != MODEL.sha256
            or profile.get("runtime_files") != libraries or profile.get("library_dirs") != ["usr/lib"]):
        raise ValueError("Referee profile does not match the acquisition pins")
    return profile


def official_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in OFFICIAL_HOSTS
            or parsed.username is not None or parsed.password is not None or parsed.port not in (None, 443)):
        raise ValueError("Download or redirect is outside the official HTTPS hosts")


class OfficialRedirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 3
    max_repeats = 1

    def __init__(self, evidence, deadline):
        self.evidence, self.deadline = evidence, deadline

    def redirect_request(self, request, response, code, message, headers, new_url):
        official_url(new_url)
        self.deadline.remaining()
        if self.evidence["http_requests"] >= MAX_HTTP_REQUESTS:
            raise ValueError("HTTP request budget reached")
        self.evidence["http_requests"] += 1
        return super().redirect_request(request, response, code, message, headers, new_url)


def download(artifact, destination, evidence, deadline, opener):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Refusing to replace an existing cached artifact")
    official_url(artifact.url)
    if evidence["http_requests"] >= MAX_HTTP_REQUESTS:
        raise ValueError("HTTP request budget reached")
    evidence["http_requests"] += 1
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        request = urllib.request.Request(artifact.url, headers={"User-Agent": "kenggo-pinned-referee-fetch/1"})
        with opener.open(request, timeout=min(15, deadline.remaining())) as response, temporary.open("wb") as target:
            official_url(response.geturl())
            if response.status != 200:
                raise ValueError(f"Unexpected HTTP status: {response.status}")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) != artifact.size):
                raise ValueError("HTTP Content-Length differs from the artifact pin")
            count = 0
            while chunk := response.read(min(1024**2, artifact.size - count + 1)):
                deadline.remaining()
                count += len(chunk)
                evidence["download_bytes"] += len(chunk)
                if count > artifact.size or evidence["download_bytes"] > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Download byte budget reached")
                target.write(chunk)
        record = verify_file(temporary, artifact)
        os.link(temporary, destination)
        evidence["artifacts"].append(dict(record, source="download"))
    finally:
        temporary.unlink(missing_ok=True)


def extract_zip_appimage(archive, destination):
    verify_file(archive, ARCHIVE)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Refusing to replace the cached AppImage")
    created = False
    try:
        with zipfile.ZipFile(archive) as source:
            members, names, total, selected = source.infolist(), set(), 0, None
            if len(members) > MAX_ZIP_ENTRIES:
                raise ValueError("ZIP member limit reached")
            for member in members:
                name = member.filename.rstrip("/")
                path = PurePosixPath(name)
                mode = member.external_attr >> 16
                if (not name or "\\" in name or path.is_absolute() or ".." in path.parts
                        or str(path) != name or name in names or member.flag_bits & 1
                        or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError("Unsafe, duplicate or encrypted ZIP member")
                names.add(name)
                total += member.file_size
                if total > MAX_ZIP_EXPANDED_BYTES:
                    raise ValueError("Expanded ZIP byte limit reached")
                if member.filename == APPIMAGE.name and not member.is_dir():
                    selected = member
            if selected is None or selected.file_size != APPIMAGE.size:
                raise ValueError("Pinned AppImage member is missing or has the wrong size")
            with source.open(selected) as data, destination.open("xb") as output:
                created = True
                count = 0
                while chunk := data.read(1024**2):
                    count += len(chunk)
                    if count > APPIMAGE.size:
                        raise ValueError("AppImage expansion limit reached")
                    output.write(chunk)
        verify_file(destination, APPIMAGE)
        destination.chmod(0o755)
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


def check_appimage(path):
    verify_file(path, APPIMAGE)
    with Path(path).open("rb") as stream:
        header = stream.read(11)
        stream.seek(APPIMAGE_OFFSET)
        filesystem = stream.read(96)
    if (header[:4] != b"\x7fELF" or header[8:11] != b"AI\x02" or len(filesystem) != 96
            or filesystem[:4] != b"hsqs" or struct.unpack_from("<HH", filesystem, 28) != (4, 0)
            or not 96 <= struct.unpack_from("<Q", filesystem, 40)[0] <= APPIMAGE.size - APPIMAGE_OFFSET):
        raise ValueError("Pinned AppImage does not have the expected SquashFS layout")


def extract_file(appimage, destination, artifact, extractor, deadline):
    if destination.exists() or destination.is_symlink():
        raise ValueError("Refusing to replace an extracted artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [str(extractor), "-cat", "-no-progress", "-processors", "1", "-offset", str(APPIMAGE_OFFSET),
               str(Path(appimage).resolve()), artifact.name]

    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (MAX_EXTRACT_CPU_SECONDS, MAX_EXTRACT_CPU_SECONDS))
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    environment = dict(os.environ)
    environment.pop("LD_PRELOAD", None)
    environment.pop("LD_LIBRARY_PATH", None)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
                                start_new_session=True, preexec_fn=limits)
    started, count, stderr_count, created = time.monotonic(), 0, 0, False
    stderr_hash = hashlib.sha256()
    try:
        with destination.open("xb") as target, selectors.DefaultSelector() as selector:
            created = True
            selector.register(process.stdout, selectors.EVENT_READ, "output")
            selector.register(process.stderr, selectors.EVENT_READ, "error")
            while selector.get_map():
                remaining = min(deadline.remaining(), MAX_EXTRACT_WALL_SECONDS - (time.monotonic() - started))
                if remaining <= 0:
                    raise TimeoutError("SquashFS extraction wall limit reached")
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    elif key.data == "output":
                        count += len(chunk)
                        if count > artifact.size:
                            raise ValueError("Extracted artifact exceeds its pinned size")
                        target.write(chunk)
                    else:
                        stderr_count += len(chunk)
                        stderr_hash.update(chunk)
                        if stderr_count > MAX_STDERR_BYTES:
                            raise ValueError("SquashFS diagnostic byte limit reached")
        if process.wait(timeout=min(deadline.remaining(), 1)) != 0:
            raise ValueError("SquashFS extraction failed")
        record = verify_file(destination, artifact)
        destination.chmod(0o755 if artifact.name in ("usr/bin/katago", "AppRun") else 0o644)
        return dict(record, source="extracted", stderr_bytes=stderr_count, stderr_sha256=stderr_hash.hexdigest())
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        process.stdout.close()
        process.stderr.close()


def verify_runtime(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Runtime directory is absent or is a symlink")
    records = [dict(verify_file(safe_path(directory, item.name), item), source="cache") for item in RUNTIME]
    expected = {item.name for item in RUNTIME if item.name.startswith("usr/lib/")}
    library_dir = safe_path(directory, "usr/lib")
    actual = {path.relative_to(directory).as_posix() for path in library_dir.rglob("*")}
    if actual != expected:
        raise ValueError("Runtime library directory contains unpinned entries")
    if not os.access(directory / "usr/bin/katago", os.X_OK):
        raise ValueError("Cached referee executable is not executable")
    return records


def acquire(cache, profile_path, profile_sha256, *, verify_only=False, evidence=None, opener=None, extract=extract_file):
    evidence = evidence if evidence is not None else {}
    evidence.update(schema="katago_referee_acquisition_v1", ok=False, verify_only=verify_only,
                    profile_sha256=profile_sha256, provenance=PROVENANCE, http_requests=0,
                    download_bytes=0, artifacts=[], gpu_seconds=0,
                    limits={"wall_seconds": MAX_WALL_SECONDS, "http_requests": MAX_HTTP_REQUESTS,
                            "download_bytes": MAX_DOWNLOAD_BYTES, "zip_entries": MAX_ZIP_ENTRIES,
                            "zip_expanded_bytes": MAX_ZIP_EXPANDED_BYTES,
                            "per_extraction_cpu_seconds": MAX_EXTRACT_CPU_SECONDS,
                            "per_extraction_wall_seconds": MAX_EXTRACT_WALL_SECONDS,
                            "per_extraction_stderr_bytes": MAX_STDERR_BYTES})
    deadline = Deadline()
    profile = validate_profile(profile_path, profile_sha256)
    evidence["profile_id"] = profile["profile_id"]
    cache = Path(cache).resolve()
    evidence["cache"] = str(cache)
    if not cache.is_dir():
        if verify_only:
            raise ValueError("Referee cache is absent")
        cache.mkdir(parents=True)
    archive, appimage, model = (safe_path(cache, item.name) for item in (ARCHIVE, APPIMAGE, MODEL))
    runtime = safe_path(cache, "squashfs-root")
    for artifact, path in ((ARCHIVE, archive), (APPIMAGE, appimage), (MODEL, model)):
        if path.exists():
            evidence["artifacts"].append(dict(verify_file(path, artifact), source="cache"))
    runtime_present = runtime.exists()
    if runtime_present:
        runtime_records = verify_runtime(runtime)
        evidence["artifacts"].extend(runtime_records)
    if verify_only:
        if not model.exists() or not runtime_present:
            raise ValueError("Required referee model or runtime is missing")
        evidence["ok"] = True
        return evidence
    if opener is None:
        opener = urllib.request.build_opener(OfficialRedirects(evidence, deadline))
    if not model.exists():
        download(MODEL, model, evidence, deadline, opener)
    if not runtime_present:
        if not appimage.exists():
            if not archive.exists():
                download(ARCHIVE, archive, evidence, deadline, opener)
            with tempfile.TemporaryDirectory(dir=cache, prefix=".appimage-") as temporary:
                candidate = Path(temporary) / APPIMAGE.name
                extract_zip_appimage(archive, candidate)
                os.link(candidate, appimage)
            evidence["artifacts"].append(dict(verify_file(appimage, APPIMAGE), source="extracted"))
        check_appimage(appimage)
        extractor = shutil.which("unsquashfs")
        if extractor is None:
            raise ValueError("Fresh extraction requires unsquashfs from squashfs-tools")
        evidence["extractor"] = {"path": extractor, "sha256": sha256(extractor), "appimage_offset": APPIMAGE_OFFSET}
        with tempfile.TemporaryDirectory(dir=cache, prefix=".runtime-") as temporary:
            candidate = Path(temporary) / "squashfs-root"
            candidate.mkdir()
            records = [extract(appimage, safe_path(candidate, artifact.name), artifact, extractor, deadline)
                       for artifact in RUNTIME]
            verify_runtime(candidate)
            if runtime.exists() or runtime.is_symlink():
                raise ValueError("Runtime appeared during extraction; existing cache is preserved")
            candidate.rename(runtime)
            evidence["artifacts"].extend(records)
    deadline.remaining()
    evidence["ok"] = True
    return evidence


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=root / ".cache" / "katago-referee")
    parser.add_argument("--profile", type=Path, default=root / "manifests" / "e1_katago_referee_profile_v1.json")
    parser.add_argument("--profile-sha256", default=PROFILE_SHA256, help="external profile-file SHA-256 pin")
    parser.add_argument("--verify-only", action="store_true", help="verify required cached files; no requests or writes")
    args = parser.parse_args()
    evidence, started = {}, time.monotonic()
    parent_cpu = time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)

    def expired(*_):
        raise TimeoutError("Acquisition wall limit reached")

    prior = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, MAX_WALL_SECONDS)
    try:
        acquire(args.cache_dir, args.profile, args.profile_sha256, verify_only=args.verify_only, evidence=evidence)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        evidence.update(ok=False, error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        evidence.update(wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - parent_cpu,
                        child_cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                        tool_sha256=sha256(Path(__file__)))
        print(json.dumps(evidence, sort_keys=True))
    return 0 if evidence.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
