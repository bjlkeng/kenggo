import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from kenggo.kgs_acquire import Client, cached_page


class Response(io.BytesIO):
    def __init__(self, body=b"ok", status=200, headers=None):
        super().__init__(body)
        self.code, self.headers = status, headers or {}


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.client = Client(Path(self.directory.name))
        self.addCleanup(self.client.log.close)
        self.client.opener = Mock()

    def records(self):
        return [json.loads(line) for line in
                (Path(self.directory.name) / "requests.jsonl").read_text().splitlines()]

    def test_rate_limit_includes_redirects(self):
        now = [100.0]
        self.client.started = now[0]
        self.client.opener.open.side_effect = [
            Response(status=302, headers={"Location": "/archives.jsp"}), Response()]
        def sleep(seconds):
            now[0] += seconds
        with patch("kenggo.kgs_acquire.time.monotonic", side_effect=lambda: now[0]), \
                patch("kenggo.kgs_acquire.time.sleep", side_effect=sleep):
            self.client.get("https://www.gokgs.com/login.jsp", data=b"user=u&password=secret")
        self.assertEqual([r["start_offset_seconds"] for r in self.records()], [0, 5])
        self.assertNotIn("secret", (Path(self.directory.name) / "requests.jsonl").read_text())
        self.assertIsNone(self.client.opener.open.call_args_list[1].args[0].data)

    def test_throttle_does_not_retry(self):
        self.client.opener.open.return_value = Response(status=429, headers={"Retry-After": "60"})
        with self.assertRaisesRegex(RuntimeError, "429"):
            self.client.get("https://www.gokgs.com/archives.jsp")
        self.assertEqual(self.client.opener.open.call_count, 1)
        self.assertEqual(self.records()[0]["status"], 429)

    def test_timeout_is_recorded_without_retry(self):
        self.client.opener.open.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            self.client.get("https://www.gokgs.com/archives.jsp")
        self.assertEqual(self.records()[0]["error"], "TimeoutError")
        self.assertEqual(self.client.opener.open.call_count, 1)

    def test_external_redirect_is_not_followed(self):
        self.client.opener.open.return_value = Response(status=302, headers={"Location": "https://example.com/"})
        with self.assertRaises(ValueError):
            self.client.get("https://www.gokgs.com/archives.jsp")
        self.assertEqual(self.client.opener.open.call_count, 1)

    def test_limits_stop_before_next_request(self):
        self.client.count = 500
        with self.assertRaises(RuntimeError):
            self.client.get("https://www.gokgs.com/archives.jsp")
        self.client.opener.open.assert_not_called()

    def test_oversized_response_stops(self):
        self.client.opener.open.return_value = Response(b"oversized")
        with self.assertRaisesRegex(RuntimeError, "byte budget"):
            self.client.get("https://www.gokgs.com/archives.jsp", limit=2)
        self.assertEqual(self.records()[0]["bytes"], 3)

    def test_prior_attempt_counts_consume_the_same_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Client(Path(directory), {"requests": 500, "response_bytes": 3007})
            self.addCleanup(client.log.close)
            client.opener = Mock()
            with self.assertRaises(RuntimeError):
                client.get("https://www.gokgs.com/archives.jsp")
            client.opener.open.assert_not_called()

    def test_cached_public_page_is_verified_without_a_request(self):
        directory = Path(self.directory.name)
        public = "Public rows.\r\nYou are currently logged in as [redacted]."
        path = directory / "sample.html"
        path.write_text(public)
        url = "https://www.gokgs.com/gameArchives.jsp?user=sample"
        (directory / "sample.json").write_text(json.dumps(dict(
            url=url, body_sha256="a" * 64, public_html_sha256=hashlib.sha256(public.encode()).hexdigest())))
        self.assertEqual(cached_page(directory, "sample", url), (public, "a" * 64))
        path.write_text("Changed rows")
        with self.assertRaises(ValueError):
            cached_page(directory, "sample", url)
        self.client.opener.open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
