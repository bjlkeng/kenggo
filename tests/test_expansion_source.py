from collections import Counter
from datetime import datetime, timedelta, timezone
import io
import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

from kenggo import expansion_source as source
from kenggo import cached_coverage_audit as cached
from test_cached_coverage_audit import Fixture as CachedFixture, row, sgf_bytes, member


class Clock:
    def __init__(self):
        self.seconds = 0
        self.origin = datetime(2026, 11, 14, tzinfo=timezone.utc)
        self.waits = []

    def now(self):
        return self.origin + timedelta(seconds=self.seconds)

    def monotonic(self):
        return self.seconds

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.seconds += seconds

    def kwargs(self):
        return dict(now=self.now, monotonic=self.monotonic, cpu=lambda: 0, sleep=self.sleep)


def html_page(item):
    import html
    return ('<table><tr><td><a href="' + item['url'] + '">Yes</a></td>' + ''.join(
        '<td>' + html.escape(item[field]) + '</td>' for field in
        ('white', 'black', 'setup', 'start', 'type', 'result')) + '</tr></table>' +
        '<a href="' + item['bundle_url'] + '">ZIP</a>').encode()


def zip_bytes(items):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for item in items:
            archive.writestr(member(item), sgf_bytes(item))
    return buffer.getvalue()


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.cached = CachedFixture(root, rows=[row('cached', black='Alice')])
        with patch.object(cached, 'source_identity', return_value={'commit': 'fixture', 'patch_sha256': '0' * 64}):
            self.old = self.cached.freeze()
        self.groups = self.cached.request['base_groups']
        self.inputs = dict(self.cached.request['inputs'])
        del self.inputs['environment']
        del self.inputs['final_quarantine']
        discovery = {'seed': source.kgs_source.SEED, 'accounts': [
            {'user': 'Alice', 'months': [1, 4, 7, 10]},
            {'user': 'Valid', 'months': [2, 5, 8, 11]},
            {'user': 'Final', 'months': [3, 6, 9, 12]},
            {'user': 'libertybot', 'months': [1, 4, 7, 10]}],
            'excluded_discovery_accounts': {'libertybot': {'reason': 'historical_bot'}}}
        prior = {'groups': sorted(self.groups.values()),
                 'source_ids': [item['archive_row']['url'] for item in self.old['candidates']],
                 'sgf_hashes': ['a' * 64], 'sequence_hashes': ['b' * 64],
                 'account_months': [['alice', '2025-01']]}
        for name, data in [('discovery_config', discovery), ('prior_exposure', prior)]:
            path = self.root / (name + '.json')
            path.write_bytes(source.encoded(data))
            self.inputs[name] = source.reference(path)
        self.inputs['old_coverage_frozen'] = source.reference(self.cached.frozen_path)
        self.request = {'schema': source.REQUEST_SCHEMA, 'refs': self.inputs,
            'cache_directory': str(self.cached.cache), 'base_groups': self.groups,
            'scope_decision': 'Synthetic source expansion test; no real network or data.',
            'deadline_utc': '2026-11-15T00:00:00Z', 'limits': source.asdict(source.Limits()),
            'source_revision': {'start_commit': 'fixture', 'code_sha256': {}}}
        self.request_path = self.root / 'collection.json'
        self.queries_path = self.root / 'queries.json'
        self.output = self.root / 'acquired'
        self.clock = Clock()

    def freeze(self):
        self.request_path.write_bytes(source.encoded(self.request))
        return source.freeze_queries(self.request_path, source.file_hash(self.request_path),
                                     self.queries_path, fixture_only=True)

    def run(self, transport):
        frozen = self.freeze()
        result = source.acquire(self.queries_path, frozen['sha256'], self.output, fixture_only=True,
                                transport=transport, **self.clock.kwargs())
        return json.loads(Path(result['path']).read_text())


class RequestTests(unittest.TestCase):
    def client(self, parent, transport, limits=None, clock=None):
        clock = clock or Clock()
        client = source.Client(parent, limits or source.Limits(), '2026-11-15T00:00:00Z',
                               transport=transport, **clock.kwargs())
        self.addCleanup(client.log.close)
        return client, clock

    def test_retries_are_charged_and_spaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            replies = iter([(503, {}, b'busy'), (429, {'Retry-After': '700'}, b'wait'), (200, {}, b'ok')])
            client, clock = self.client(tmp, lambda *args: next(replies))
            self.assertEqual(client.get('https://www.gokgs.com/gameArchives.jsp', 100), b'ok')
            self.assertEqual(client.requests, 3)
            self.assertEqual(client.response_bytes, 10)
            self.assertEqual(clock.waits, [300, 0, 700, 0])
            records = [json.loads(line) for line in Path(tmp, 'requests.jsonl').read_text().splitlines()]
            self.assertEqual([r['status'] for r in records], [503, 429, 200])
            self.assertGreaterEqual(records[1]['start_offset_seconds'] - records[0]['start_offset_seconds'], 5)

    def test_login_uses_private_file_and_charges_every_request_without_logging_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            secret = Path(tmp, 'credentials.json')
            secret.write_text(json.dumps({'username': 'private-user', 'password': 'private-password'}))
            secret.chmod(0o600)
            item = row('new', black='Alice')
            item.update(page_key='Alice-2025-01', bundle_url='https://www.gokgs.com/servlet/archives/Alice-2025-01.zip')
            login = b'<form action="login.jsp" method="post"><input name="user"><input name="password" type="password"></form>'
            calls = []
            def transport(url, timeout, maximum, **kwargs):
                calls.append((url, kwargs))
                if len(calls) == 1:
                    return 200, {}, login
                if len(calls) == 2:
                    return 200, {}, b'logged in'
                return 200, {}, html_page(item)
            client, clock = self.client(tmp, transport)
            query = {'url': 'https://www.gokgs.com/gameArchives.jsp?user=Alice&year=2025&month=1',
                     'user': 'Alice', 'month': 1, 'key': 'Alice-2025-01'}
            self.assertEqual(client.authenticate(secret, query), html_page(item))
            self.assertEqual(client.requests, 3)
            self.assertEqual(client.response_bytes, len(login) + len(b'logged in') + len(html_page(item)))
            self.assertEqual(clock.seconds, 10)
            self.assertEqual(calls[1], ('https://www.gokgs.com/login.jsp',
                {'data': b'user=private-user&password=private-password'}))
            ledger = Path(tmp, 'requests.jsonl').read_text()
            self.assertNotIn('private-user', ledger)
            self.assertNotIn('private-password', ledger)
            self.assertEqual([json.loads(line)['method'] for line in ledger.splitlines()], ['GET', 'POST', 'GET'])

    def test_login_never_posts_to_an_untrusted_form_or_forwards_a_post_redirect(self):
        for attack in ('form', 'redirect'):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as tmp:
                secret = Path(tmp, 'credentials.json')
                secret.write_text(json.dumps({'username': 'user', 'password': 'secret'})); secret.chmod(0o600)
                login = ('<form action="' + ('https://other.example/login' if attack == 'form' else 'login.jsp') +
                         '" method="post"><input name="user"><input name="password"></form>').encode()
                calls = []
                def transport(url, *args, **kwargs):
                    calls.append(url)
                    return (200, {}, login) if len(calls) == 1 else (307, {'Location': '/gameArchives.jsp'}, b'')
                client, _ = self.client(tmp, transport)
                with self.assertRaises(source.AuthenticationRequired):
                    client.authenticate(secret, {'url': 'https://www.gokgs.com/gameArchives.jsp'})
                self.assertEqual(len(calls), 1 if attack == 'form' else 2)

    def test_public_credential_file_is_rejected_before_network_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'credentials.json'); path.write_text('{}'); path.chmod(0o644)
            calls = []
            client, _ = self.client(tmp, lambda *args: calls.append(args))
            with self.assertRaisesRegex(ValueError, 'owner-only'):
                client.authenticate(path, {'url': 'https://www.gokgs.com/gameArchives.jsp'})
            self.assertEqual(calls, [])

    def test_timeout_retries_stop_without_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            def timeout(*args):
                raise TimeoutError('fixture')
            client, _ = self.client(tmp, timeout)
            with self.assertRaisesRegex(source.RequestFailure, 'retries_exhausted'):
                client.get('https://www.gokgs.com/gameArchives.jsp', 100)
            self.assertEqual(client.requests, 3)
            self.assertEqual(client.response_bytes, 0)

    def test_request_byte_and_absolute_deadline_caps(self):
        for kind in ('requests', 'bytes', 'deadline'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                calls = []
                def transport(*args):
                    calls.append(args)
                    return 200, {}, b'12345'
                limits = source.Limits(requests=1) if kind == 'requests' else source.Limits(response_bytes=4) if kind == 'bytes' else source.Limits()
                clock = Clock()
                if kind == 'deadline':
                    clock.seconds = 86400
                client, _ = self.client(tmp, transport, limits, clock)
                if kind == 'requests':
                    client.get('https://www.gokgs.com/gameArchives.jsp', 100)
                with self.assertRaises(source.SourceStop):
                    client.get('https://www.gokgs.com/gameArchives.jsp', 100)
                self.assertEqual(len(calls), 0 if kind == 'deadline' else 1)

    def test_redirect_cannot_fetch_sgf_during_metadata_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def transport(url, *args):
                calls.append(url)
                return 302, {'Location': 'https://files.gokgs.com/games/2025/1/1/secret.sgf'}, b''
            client, _ = self.client(tmp, transport)
            with self.assertRaisesRegex(ValueError, 'request type'):
                client.get('https://www.gokgs.com/gameArchives.jsp', 100)
            self.assertEqual(len(calls), 1)

    def test_redirects_are_requests_and_get_five_second_spacing(self):
        with tempfile.TemporaryDirectory() as tmp:
            replies = iter([(302, {'Location': '/archives.jsp'}, b'first'), (200, {}, b'last')])
            client, clock = self.client(tmp, lambda *args: next(replies))
            self.assertEqual(client.get('https://www.gokgs.com/gameArchives.jsp', 100), b'last')
            self.assertEqual(client.requests, 2)
            self.assertEqual(client.response_bytes, 9)
            self.assertEqual(clock.seconds, 5)

    def test_real_http_reader_handles_success_and_http_error_without_network(self):
        class Socket:
            def __init__(self, data):
                self.data, self.timeouts = data, []

            def makefile(self, *args):
                raw = io.BytesIO(self.data)
                raw._sock = self
                return io.BufferedReader(raw)

            def settimeout(self, value):
                self.timeouts.append(value)

        for status in (200, 503):
            with self.subTest(status=status):
                sock = Socket(f'HTTP/1.1 {status} Fixture\r\nContent-Length: 3\r\n\r\nabc'.encode())
                wire = http.client.HTTPResponse(sock)
                wire.begin()
                wire.code = wire.status
                with patch.object(source.urllib.request, 'build_opener') as build:
                    if status == 200:
                        build.return_value.open.return_value = wire
                    else:
                        build.return_value.open.side_effect = urllib.error.HTTPError(
                            'https://www.gokgs.com/archives.jsp', status, 'fixture', wire.headers, wire)
                    actual_status, _, body = source.http_response('https://www.gokgs.com/archives.jsp', 30, 100)
                self.assertEqual((actual_status, body), (status, b'abc'))
                self.assertTrue(sock.timeouts and all(0 < value <= 30 for value in sock.timeouts))

    def test_premature_eof_is_an_interrupted_response_with_received_bytes(self):
        class Socket:
            def makefile(self, *args):
                raw = io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Length: 1003\r\n\r\nabc')
                raw._sock = self
                return io.BufferedReader(raw)

            def settimeout(self, value):
                pass

        wire = http.client.HTTPResponse(Socket())
        wire.begin()
        wire.code = wire.status
        with patch.object(source.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = wire
            with self.assertRaises(source.InterruptedResponse) as caught:
                source.http_response('https://www.gokgs.com/archives.jsp', 30, 2000)
        self.assertEqual(caught.exception.body, b'abc')
        self.assertEqual(caught.exception.status, 200)

    def test_partial_response_bytes_are_charged_before_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def transport(*args):
                calls.append(args)
                if len(calls) == 1:
                    raise source.InterruptedResponse(200, {}, b'part', 'TimeoutError')
                return 200, {}, b'whole'
            client, _ = self.client(tmp, transport)
            self.assertEqual(client.get('https://www.gokgs.com/archives.jsp', 100), b'whole')
            self.assertEqual(client.response_bytes, 9)
            self.assertEqual(client.requests, 2)

    def test_http_date_retry_after_and_wall_limit_are_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            replies = iter([(503, {'Retry-After': 'Sat, 14 Nov 2026 00:10:00 GMT'}, b''), (200, {}, b'ok')])
            client, clock = self.client(tmp, lambda *args: next(replies))
            self.assertEqual(client.get('https://www.gokgs.com/archives.jsp', 100), b'ok')
            self.assertEqual(clock.seconds, 600)
        with tempfile.TemporaryDirectory() as tmp:
            client, _ = self.client(tmp, lambda *args: (503, {}, b''), source.Limits(wall_seconds=200))
            with self.assertRaisesRegex(source.SourceStop, 'insufficient_time'):
                client.get('https://www.gokgs.com/archives.jsp', 100)
            self.assertEqual(client.requests, 1)


class ExpansionTests(unittest.TestCase):
    def test_continuation_keeps_the_current_groups_and_skips_completed_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            f.freeze()
            frozen = json.loads(f.queries_path.read_text())
            _, objects, _, _ = source._context(frozen['request'], fixture_only=True)
            old = objects['old_coverage_frozen']['candidates'][0]
            groups = {**objects['old_coverage_frozen']['groups'], 'newalias': 'kgs-handle:newalias'}
            objects.update(current_groups=groups, completed_ids={old['archive_row']['url']})
            fresh = row('fresh', black='Alice')
            fresh.update(page_key='Alice-2025-01', bundle_url='https://www.gokgs.com/servlet/archives/Alice-2025-01.zip')
            result = source.freeze_candidates(frozen, f.request, objects, [old['archive_row'], fresh])
            self.assertEqual(result['groups']['newalias'], 'kgs-handle:newalias')
            self.assertEqual([item['archive_row']['url'] for item in result['candidates']], [fresh['url']])

    def test_freeze_queries_excludes_final_prefix_and_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            with patch.object(zipfile.ZipFile, 'open', side_effect=AssertionError('SGF opened')):
                fixture.freeze()
            frozen = json.loads(fixture.queries_path.read_text())
            queries = frozen['queries']
            self.assertEqual(Counter(item['user'] for item in queries), {'Alice': 10, 'Valid': 2})
            self.assertEqual({item['month'] for item in queries if item['user'] == 'Valid'}, {9, 10})
            self.assertEqual(len(frozen['old_candidates']), 1)

    def test_metadata_is_frozen_before_new_bundle_and_members_stay_unopened(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            new = row('extra', 2, black='Alice', page_key='Alice-2025-02',
                      bundle_url='https://www.gokgs.com/servlet/archives/en_US/Alice-all-2025-2.zip')
            forbidden = row('reserved', 2, black='Alice', white='Final', page_key='Alice-2025-02', bundle_url=new['bundle_url'])
            payload = zip_bytes([new, forbidden])
            requests = []
            def transport(url, timeout, maximum):
                requests.append(url)
                if url.endswith('.zip'):
                    selected = json.loads((fixture.output / 'frozen_candidates.json').read_text())
                    self.assertEqual(len(selected['candidates']), 2)
                    self.assertNotIn(forbidden['url'], [r['archive_row']['url'] for r in selected['candidates']])
                    return 200, {}, payload
                if 'user=Alice' in url and 'month=2&' in url:
                    body = html_page(new).replace(b'</table>', html_page(forbidden).split(b'<table>')[1].split(b'</table>')[0] + b'</table>')
                    return 200, {}, body
                return 200, {}, b'No games'
            with patch.object(zipfile.ZipFile, 'open', side_effect=AssertionError('source stage opened SGF')):
                result = fixture.run(transport)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(len(result['candidates']), 2)
            self.assertTrue(all('sgf_sha256' not in item['source'] for item in result['candidates']))
            self.assertEqual(sum(url.endswith('.zip') for url in requests), 1)
            self.assertIn('a' * 64, result['exposure']['sgf_hashes'])
            self.assertIn(new['url'], result['exposure']['source_ids'])
            self.assertNotIn('kgs-handle:final', {g for item in result['candidates'] for _, g in cached.participants(item['archive_row'], result['groups'])})

    def test_login_block_retains_cached_candidates_without_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            calls = []
            def transport(url, *args):
                calls.append(url)
                return 200, {}, b'<form><input name="password"></form>'
            result = fixture.run(transport)
            self.assertEqual(result['status'], 'authentication_blocked')
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(result['candidates']), 1)
            self.assertEqual(result['counts']['query_statuses'], {'failed': 1, 'unattempted': 11})
            progress = json.loads((fixture.output / 'progress.json').read_text())
            self.assertEqual(progress['phase'], 'authentication_blocked')
            self.assertEqual(progress['queries_recorded'], 12)
            self.assertEqual(progress['bundles_indexed'], 1)
            self.assertEqual(progress['costs']['requests'], 1)
            self.assertNotIn('candidates', progress)
            with self.assertRaises(FileExistsError):
                source.acquire(fixture.queries_path, source.file_hash(fixture.queries_path),
                    fixture.root / 'repeated-acquisition', fixture_only=True,
                    transport=transport, **fixture.clock.kwargs())
            self.assertEqual(len(calls), 1)

    def test_changed_inputs_and_lost_exposure_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            frozen = fixture.freeze()
            fixture.cached.cache.joinpath(fixture.cached.rows[0]['page_key'] + '.html').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'HTML changed'):
                source.acquire(fixture.queries_path, frozen['sha256'], fixture.output, fixture_only=True)
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            path = Path(fixture.inputs['prior_exposure']['path'])
            data = json.loads(path.read_text())
            data['source_ids'] = []
            path.write_bytes(source.encoded(data))
            fixture.request['refs']['prior_exposure'] = source.reference(path)
            with self.assertRaisesRegex(ValueError, 'previously inspected'):
                fixture.freeze()

    def test_selection_preserves_cached_ids_and_balances_rank_and_account(self):
        items = []
        for account, rank, count in [('Alice', '4k', 8), ('Bob', '4k', 2), ('Carol', '15k', 2)]:
            for index in range(count):
                item = row(account + str(index), black=account, white='Other',
                           page_key=account + '-2025-01')
                item['black'] = account + ' [' + rank + ']'
                item['white'] = 'Other [' + rank + ']'
                items.append({'archive_row': item, 'provisional_role': 'train'})
        selected = source.balanced(items, 4)
        self.assertEqual(selected, source.balanced(list(reversed(items)), 4))
        self.assertEqual(Counter(item['archive_row']['black'].split()[1] for item in selected), {'[4k]': 2, '[15k]': 2})
        self.assertEqual(len({item['archive_row']['black'] for item in selected}), 3)
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            fixture.request['limits']['max_train_candidates'] = 1
            fixture.freeze()
            frozen = json.loads(fixture.queries_path.read_text())
            request, objects, *_ = source._context(frozen['request'], fixture_only=True)
            result = source.freeze_candidates(frozen, request, objects, [item['archive_row'] for item in items])
            self.assertEqual([item['archive_row']['url'] for item in result['candidates']],
                             [fixture.cached.rows[0]['url']])

    def test_only_unique_zip_member_is_made_accessible(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            new = row('missing', 2, black='Alice', page_key='Alice-2025-02',
                      bundle_url='https://www.gokgs.com/servlet/archives/en_US/Alice-all-2025-2.zip')
            def transport(url, *args):
                if url.endswith('.zip'):
                    return 200, {}, zip_bytes([row('other', 2, black='Alice')])
                if 'user=Alice' in url and 'month=2&' in url:
                    return 200, {}, html_page(new)
                return 200, {}, b'No games'
            result = fixture.run(transport)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(len(result['candidates']), 1)
            self.assertEqual(result['counts']['frozen_candidates'], 2)
            self.assertIn(new['url'], result['exposure']['source_ids'])
            self.assertTrue(any(d.get('source_url') == new['url'] for d in result['diagnostics']))

    def test_expanded_archive_limit_stops_further_bundle_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(tmp)
            with zipfile.ZipFile(fixture.cached.cache / (fixture.cached.rows[0]['page_key'] + '.zip')) as archive:
                old_size = sum(info.file_size for info in archive.infolist())
            fixture.request['limits']['expanded_zip_bytes'] = old_size + 1
            new = row('large', 2, black='Alice', page_key='Alice-2025-02',
                      bundle_url='https://www.gokgs.com/servlet/archives/en_US/Alice-all-2025-2.zip')
            later = row('later', 3, black='Alice', page_key='Alice-2025-03',
                        bundle_url='https://www.gokgs.com/servlet/archives/en_US/Alice-all-2025-3.zip')
            bundles = []
            def transport(url, *args):
                if url.endswith('.zip'):
                    bundles.append(url)
                    return 200, {}, zip_bytes([new])
                for item, month in ((new, 2), (later, 3)):
                    if 'user=Alice' in url and f'month={month}&' in url:
                        return 200, {}, html_page(item)
                return 200, {}, b'No games'
            result = fixture.run(transport)
            self.assertEqual(len(bundles), 1)
            self.assertEqual(result['stop_reason'], 'expanded_zip_limit')
            self.assertEqual(result['counts']['frozen_candidates'], 3)
            self.assertEqual(len(result['candidates']), 1)
            self.assertEqual(result['counts']['expanded_zip_bytes'], old_size)


if __name__ == '__main__':
    unittest.main()
