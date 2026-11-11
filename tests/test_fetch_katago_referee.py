"""Pinned acquisition behavior using tiny temporary artifacts and fake HTTP."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.request
import zipfile

from tools import fetch_katago_referee as fetch


def hashed(data):
    return hashlib.sha256(data).hexdigest()


class Response(io.BytesIO):
    def __init__(self, body, url, *, status=200, length=True):
        super().__init__(body)
        self.url, self.status = url, status
        self.headers = {"Content-Length": str(len(body))} if length else {}

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, bodies):
        self.bodies, self.requests = bodies, []

    def open(self, request, timeout):
        self.requests.append((request.full_url, timeout))
        result = self.bodies[request.full_url]
        return result() if callable(result) else Response(result, request.full_url)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.runtime_bodies = {"usr/bin/katago": b"engine", "usr/lib/libbz2.so.1.0": b"bz2",
                               "usr/lib/libz.so.1": b"zlib", "usr/lib/libzip.so.4": b"zip", "AppRun": b"wrapper"}
        self.image = bytearray(224)
        self.image[:11] = b"\x7fELF\x02\x01\x01\x00AI\x02"
        self.image[128:132] = b"hsqs"
        struct.pack_into("<HH", self.image, 128 + 28, 4, 0)
        struct.pack_into("<Q", self.image, 128 + 40, 96)
        self.image = bytes(self.image)
        self.zip_bytes = self.zip_data()
        archive = fetch.Artifact("engine.zip", "https://github.com/lightvector/KataGo/fixture.zip",
                                 len(self.zip_bytes), hashed(self.zip_bytes))
        appimage = fetch.Artifact("katago", archive.url + "#katago", len(self.image), hashed(self.image))
        model = fetch.Artifact("model.txt.gz", "https://media.katagotraining.org/fixture.gz", 5, hashed(b"model"))
        runtime = tuple(fetch.Artifact(name, appimage.url, len(body), hashed(body))
                        for name, body in self.runtime_bodies.items())
        changes = patch.multiple(fetch, ARCHIVE=archive, APPIMAGE=appimage, MODEL=model,
                                 RUNTIME=runtime, APPIMAGE_OFFSET=128)
        changes.start()
        self.addCleanup(changes.stop)
        profile = {"schema": "katago_referee_profile_v1", "profile_id": "acquisition-fixture",
                   "artifacts": {"engine": runtime[0].sha256, "model": model.sha256},
                   "runtime_files": {item.name: item.sha256 for item in runtime if item.name.startswith("usr/lib/")},
                   "library_dirs": ["usr/lib"]}
        self.profile = self.root / "profile.json"
        self.profile.write_text(json.dumps(profile))
        self.profile_sha256 = fetch.sha256(self.profile)
        self.extractor = self.root / "unsquashfs"
        self.extractor.write_text("fake extractor; never executed by the fake extraction tests")

    def zip_data(self, extra=()):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            item = zipfile.ZipInfo("katago")
            item.external_attr = (stat.S_IFREG | 0o755) << 16
            archive.writestr(item, self.image)
            for item, data in extra:
                archive.writestr(item, data)
        return stream.getvalue()

    def populate(self, *, packages=True):
        (self.cache / fetch.MODEL.name).write_bytes(b"model")
        if packages:
            (self.cache / fetch.ARCHIVE.name).write_bytes(self.zip_bytes)
            (self.cache / fetch.APPIMAGE.name).write_bytes(self.image)
        runtime = self.cache / "squashfs-root"
        for name, body in self.runtime_bodies.items():
            path = runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            if name in ("usr/bin/katago", "AppRun"):
                path.chmod(0o755)

    def acquire(self, **kwargs):
        return fetch.acquire(self.cache, self.profile, self.profile_sha256, **kwargs)

    def test_verify_only_and_normal_cache_reuse_make_no_requests_or_writes(self):
        self.populate()
        before = {path.relative_to(self.cache): (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.cache.rglob("*") if path.is_file()}
        opener = Opener({})
        for verify_only in (True, False):
            evidence = self.acquire(verify_only=verify_only, opener=opener)
            self.assertTrue(evidence["ok"])
            self.assertEqual((evidence["http_requests"], evidence["download_bytes"]), (0, 0))
        after = {path.relative_to(self.cache): (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in self.cache.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(opener.requests, [])

    def test_valid_runtime_does_not_require_archival_packages(self):
        self.populate(packages=False)
        self.assertTrue(self.acquire(verify_only=True, opener=Opener({}))["ok"])
        self.assertTrue(self.acquire(opener=Opener({}))["ok"])
        self.assertFalse((self.cache / fetch.ARCHIVE.name).exists())

    def test_missing_verify_only_cache_is_not_created(self):
        self.cache.rmdir()
        with self.assertRaisesRegex(ValueError, "cache is absent"):
            self.acquire(verify_only=True, opener=Opener({}))
        self.assertFalse(self.cache.exists())

    def test_download_and_extract_are_sequential_and_pinned(self):
        opener = Opener({fetch.MODEL.url: b"model", fetch.ARCHIVE.url: self.zip_bytes})
        extracted = []

        def extract(appimage, destination, artifact, extractor, deadline):
            fetch.check_appimage(appimage)
            self.assertEqual(extractor, str(self.extractor))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.runtime_bodies[artifact.name])
            destination.chmod(0o755)
            extracted.append(artifact.name)
            return dict(fetch.verify_file(destination, artifact), source="extracted")

        with patch.object(fetch.shutil, "which", return_value=str(self.extractor)), patch.object(fetch.subprocess, "Popen") as execute:
            evidence = self.acquire(opener=opener, extract=extract)
        execute.assert_not_called()
        self.assertEqual([url for url, _ in opener.requests], [fetch.MODEL.url, fetch.ARCHIVE.url])
        self.assertEqual(extracted, [item.name for item in fetch.RUNTIME])
        self.assertEqual(evidence["http_requests"], 2)
        self.assertTrue(self.acquire(verify_only=True, opener=Opener({}))["ok"])
        self.assertFalse(any(path.name.startswith(".") for path in self.cache.iterdir()))

    def test_corrupt_or_partial_cache_is_preserved_without_repair_downloads(self):
        self.populate()
        model = self.cache / fetch.MODEL.name
        model.write_bytes(b"wrong")
        opener = Opener({})
        with self.assertRaisesRegex(ValueError, "size/hash mismatch"):
            self.acquire(opener=opener)
        self.assertEqual(model.read_bytes(), b"wrong")
        model.write_bytes(b"model")
        (self.cache / "squashfs-root/usr/lib/libz.so.1").unlink()
        with self.assertRaisesRegex(ValueError, "size/hash mismatch"):
            self.acquire(opener=opener)
        self.assertEqual(opener.requests, [])

    def test_runtime_symlinks_and_extra_libraries_are_rejected(self):
        self.populate()
        unexpected = self.cache / "squashfs-root/usr/lib/unpinned.so"
        unexpected.write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "unpinned entries"):
            self.acquire(verify_only=True)
        unexpected.unlink()
        model = self.cache / fetch.MODEL.name
        model.unlink()
        external = self.root / "external-model"
        external.write_bytes(b"model")
        model.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.acquire(verify_only=True)
        self.assertEqual(external.read_bytes(), b"model")

    def test_profile_external_pin_and_runtime_pins_are_required(self):
        with self.assertRaisesRegex(ValueError, "external pin"):
            fetch.validate_profile(self.profile, "0" * 64)
        profile = json.loads(self.profile.read_text())
        profile["artifacts"]["model"] = "0" * 64
        self.profile.write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError, "acquisition pins"):
            fetch.validate_profile(self.profile, fetch.sha256(self.profile))

    def test_profile_v2_can_change_policy_without_changing_artifact_pins(self):
        profile = json.loads(self.profile.read_text())
        profile.update(schema="katago_referee_profile_v2", score_selection={"policy": "explicit-version-2"})
        self.profile.write_text(json.dumps(profile))
        self.assertEqual(fetch.validate_profile(self.profile, fetch.sha256(self.profile)), profile)
        profile["schema"] = "unknown_future_schema"
        self.profile.write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError, "acquisition pins"):
            fetch.validate_profile(self.profile, fetch.sha256(self.profile))

    def test_unsafe_zip_members_are_rejected_before_writing(self):
        symlink = zipfile.ZipInfo("link")
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        cases = [("../escape", b"x"), ("/absolute", b"x"), ("a\\b", b"x"), (symlink, b"target")]
        for item in cases:
            raw = self.zip_data([item])
            archive = self.cache / "unsafe.zip"
            archive.write_bytes(raw)
            pin = fetch.Artifact("unsafe.zip", fetch.ARCHIVE.url, len(raw), hashed(raw))
            with self.subTest(item=item[0]), patch.object(fetch, "ARCHIVE", pin), self.assertRaisesRegex(ValueError, "Unsafe"):
                fetch.extract_zip_appimage(archive, self.cache / "extracted")
            self.assertFalse((self.cache / "extracted").exists())
        self.assertFalse((self.root / "escape").exists())

    def test_archive_hash_is_checked_before_extraction(self):
        archive = self.cache / fetch.ARCHIVE.name
        archive.write_bytes(b"not the pinned archive")
        with patch.object(fetch.zipfile, "ZipFile") as open_zip, self.assertRaises(ValueError):
            fetch.extract_zip_appimage(archive, self.cache / "extracted")
        open_zip.assert_not_called()

    def test_expanded_zip_limit_is_checked_before_writing(self):
        archive = self.cache / fetch.ARCHIVE.name
        archive.write_bytes(self.zip_bytes)
        with patch.object(fetch, "MAX_ZIP_EXPANDED_BYTES", 1), self.assertRaisesRegex(ValueError, "Expanded ZIP"):
            fetch.extract_zip_appimage(archive, self.cache / "extracted")
        self.assertFalse((self.cache / "extracted").exists())

    def test_download_failures_leave_no_partial_or_final_file(self):
        cases = [lambda: Response(b"toolong", fetch.MODEL.url, length=False),
                 lambda: Response(b"wrong", fetch.MODEL.url),
                 lambda: Response(b"model", fetch.MODEL.url, status=500),
                 lambda: Response(b"model", "https://untrusted.example/model")]
        for response in cases:
            evidence = {"http_requests": 0, "download_bytes": 0, "artifacts": []}
            with self.subTest(response=response), self.assertRaises(ValueError):
                fetch.download(fetch.MODEL, self.cache / fetch.MODEL.name, evidence, fetch.Deadline(),
                               Opener({fetch.MODEL.url: response}))
            self.assertEqual(evidence["http_requests"], 1)
            self.assertEqual(list(self.cache.iterdir()), [])

    def test_redirect_request_and_deadline_limits(self):
        evidence = {"http_requests": fetch.MAX_HTTP_REQUESTS}
        handler = fetch.OfficialRedirects(evidence, fetch.Deadline())
        request = urllib.request.Request(fetch.ARCHIVE.url)
        with self.assertRaisesRegex(ValueError, "request budget"):
            handler.redirect_request(request, None, 302, "redirect", {}, fetch.ARCHIVE.url)
        with self.assertRaisesRegex(ValueError, "official HTTPS"):
            fetch.official_url("http://github.com/lightvector/KataGo")
        evidence = {"http_requests": 0, "download_bytes": 0, "artifacts": []}
        with self.assertRaises(TimeoutError):
            fetch.download(fetch.MODEL, self.cache / fetch.MODEL.name, evidence, fetch.Deadline(-1), Opener({}))
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_appimage_header_and_offset_are_verified_without_executing_it(self):
        path = self.cache / "katago"
        path.write_bytes(self.image)
        with patch.object(fetch.subprocess, "Popen") as execute:
            fetch.check_appimage(path)
        execute.assert_not_called()
        changed = b"bad!" + self.image[4:]
        path.write_bytes(changed)
        pin = fetch.Artifact("katago", fetch.APPIMAGE.url, len(changed), hashed(changed))
        with patch.object(fetch, "APPIMAGE", pin), self.assertRaisesRegex(ValueError, "SquashFS layout"):
            fetch.check_appimage(path)

    def test_main_verify_only_prints_reproducible_schema(self):
        self.populate()
        args = ["fetch", "--verify-only", "--cache-dir", str(self.cache), "--profile", str(self.profile),
                "--profile-sha256", self.profile_sha256]
        output = io.StringIO()
        with patch.object(sys, "argv", args), redirect_stdout(output):
            result = fetch.main()
        evidence = json.loads(output.getvalue())
        self.assertEqual((result, evidence["schema"], evidence["http_requests"]),
                         (0, "katago_referee_acquisition_v1", 0))
        self.assertIn("model_license_url", evidence["provenance"])


class ExtractionProcessTests(unittest.TestCase):
    def test_existing_destination_is_preserved_and_extractor_is_not_started(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "existing"
            destination.write_bytes(b"preserve")
            artifact = fetch.Artifact("example", "fixture", 3, hashed(b"abc"))
            with patch.object(fetch.subprocess, "Popen") as execute, self.assertRaisesRegex(ValueError, "replace"):
                fetch.extract_file(Path(temporary) / "image", destination, artifact, "unsquashfs", fetch.Deadline())
            execute.assert_not_called()
            self.assertEqual(destination.read_bytes(), b"preserve")

    def test_extractor_wall_limit_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "fake-unsquashfs"
            executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(1)\n")
            executable.chmod(0o755)
            artifact = fetch.Artifact("example", "fixture", 3, hashed(b"abc"))
            destination = root / "output"
            with patch.object(fetch, "MAX_EXTRACT_WALL_SECONDS", 0.03), self.assertRaises(TimeoutError):
                fetch.extract_file(root / "image", destination, artifact, executable, fetch.Deadline())
            self.assertFalse(destination.exists())

    def test_bounded_cat_output_is_verified_and_overflow_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "fake-unsquashfs"
            artifact = fetch.Artifact("usr/lib/example", "fixture", 3, hashed(b"abc"))
            for body, accepted in ((b"abc", True), (b"abcd", False)):
                executable.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.buffer.write({body!r})\n")
                executable.chmod(0o755)
                destination = root / "output"
                if accepted:
                    record = fetch.extract_file(root / "image", destination, artifact, executable, fetch.Deadline())
                    self.assertEqual(record["sha256"], artifact.sha256)
                    self.assertEqual(destination.read_bytes(), b"abc")
                    destination.unlink()
                else:
                    with self.assertRaisesRegex(ValueError, "pinned size"):
                        fetch.extract_file(root / "image", destination, artifact, executable, fetch.Deadline())
                    self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
