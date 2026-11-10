"""Deterministic discovery and sampling for the bounded KGS source audit."""

from collections import defaultdict
from datetime import date
from hashlib import sha256
from html.parser import HTMLParser
import re
from urllib.parse import urljoin, urlsplit


SEED = "kenggo-6yr.2-20261110-v1"
RANKS = [f"{n}k" for n in range(15, 0, -1)] + ["1d"]


def digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def discovery_accounts(survey: str) -> list[dict]:
    groups = defaultdict(dict)
    for line in survey.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 3:
            continue
        user, rank, month = cells
        if (rank in RANKS and re.fullmatch(r"[A-Za-z0-9]+", user)
                and re.fullmatch(r"20\d\d-\d\d", month)):
            groups[rank][user.lower()] = {"user": user, "survey_rank": rank,
                                          "survey_month": month}
    selected = []
    for rank in RANKS:
        members = sorted(groups[rank].values(), key=lambda x: digest(
            SEED + ":account:" + x["user"].lower()))[:3]
        for index, member in enumerate(members):
            selected.append({**member, "months": list(range(1 + index, 13, 3))})
    return selected


class ArchivePage(HTMLParser):
    def __init__(self, body: str):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.unlinked_rows = []
        self.links = []
        self._row = None
        self._cell = None
        self._span = 1
        self._row_links = []
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            if self._row is not None:
                self.handle_endtag("tr")
            self._row, self._row_links = [], []
        elif tag in {"td", "th"} and self._row is not None:
            if self._cell is not None:
                self.handle_endtag("td")
            self._cell = []
            self._span = int(attrs.get("colspan", "1"))
            if self._span < 1:
                raise ValueError("Invalid archive column span")
        elif tag == "a" and "href" in attrs:
            link = urljoin("https://www.gokgs.com/", attrs["href"])
            self.links.append(link)
            if self._row is not None:
                self._row_links.append(link)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._row.extend([""] * min(self._span - 1, 7))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._cell is not None:
                self.handle_endtag("td")
            sgfs = sorted(set(link for link in self._row_links if is_sgf_url(link)))
            if sgfs:
                if len(sgfs) != 1 or len(self._row) != 7:
                    raise ValueError(f"Unrecognized archive game row: {len(sgfs)} SGFs, cells={self._row!r}")
                self.rows.append(dict(zip(
                    ("viewable", "white", "black", "setup", "start", "type", "result"),
                    self._row, strict=True), url=sgfs[0]))
            elif len(self._row) == 7 and self._row[0] in {"Yes", "No"}:
                self.unlinked_rows.append(dict(zip(
                    ("viewable", "white", "black", "setup", "start", "type", "result"),
                    self._row, strict=True)))
            self._row = self._cell = None
        elif tag == "table" and self._row is not None:
            self.handle_endtag("tr")

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def is_sgf_url(url: str) -> bool:
    parsed = urlsplit(url)
    return (parsed.scheme == "https" and parsed.netloc == "files.gokgs.com"
            and re.fullmatch(r"/games/\d{4}/\d{1,2}/\d{1,2}/[^/]+\.sgf", parsed.path)
            is not None and not parsed.query and not parsed.fragment)


def player(cell: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"([A-Za-z0-9]+) \[([^\]]+)\]", cell)
    return match.groups() if match else None


def candidate_reasons(row: dict) -> list[str]:
    reasons = []
    if row["type"] != "Ranked":
        reasons.append("not_ranked")
    if row["setup"] != "19×19":
        reasons.append("not_even_19x19")
    for color in ("white", "black"):
        parsed = player(row[color])
        if parsed is None or parsed[1] not in RANKS:
            reasons.append(color + "_rank_unknown_or_out_of_scope")
    if not is_sgf_url(row["url"]):
        return reasons + ["unsupported_source_url"]
    parts = urlsplit(row["url"]).path.split("/")
    try:
        source_date = date(*(int(value) for value in parts[2:5]))
    except ValueError:
        return reasons + ["invalid_source_date"]
    if source_date.year != 2025:
        reasons.append("outside_2025")
    return reasons


def select_candidates(rows: list[dict], count: int = 1000) -> list[dict]:
    unique = {}
    for row in rows:
        if row["url"] in unique:
            previous = unique[row["url"]]
            if any(previous[k] != row[k] for k in ("white", "black", "setup", "type", "result")):
                raise ValueError("Conflicting archive metadata for one source ID")
        else:
            unique[row["url"]] = row
    strata = defaultdict(list)
    for row in unique.values():
        if candidate_reasons(row):
            continue
        rank = min((player(row[c])[1] for c in ("white", "black")), key=RANKS.index)
        month = int(urlsplit(row["url"]).path.split("/")[3])
        strata[(rank, month)].append(row)
    for group in strata.values():
        group.sort(key=lambda row: digest(SEED + ":game:" + row["url"]))
    keys = [(rank, month) for month in range(1, 13) for rank in RANKS]
    selected = []
    index = 0
    while len(selected) < count:
        added = 0
        for key in keys:
            if len(strata[key]) > index:
                selected.append(strata[key][index])
                added += 1
                if len(selected) == count:
                    break
        if not added:
            break
        index += 1
    return selected
