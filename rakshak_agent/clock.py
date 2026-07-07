"""
Time source for real-time CA reasoning.

`now` is INJECTABLE: defaults to the system date (real-time), but tests pin a
fixed date so advisory outputs stay deterministic ("given date D + same data ->
same advice"). Zero cost.
"""

from __future__ import annotations

import datetime
import re

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(20\d\d)\b",
    re.IGNORECASE)


class Clock:
    def __init__(self, today: datetime.date = None):
        self._today = today

    def today(self) -> datetime.date:
        return self._today or datetime.date.today()


def parse_date(text: str):
    m = _DATE_RE.search(text or "")
    if not m:
        return None
    try:
        return datetime.date(int(m.group(3)), _MONTHS[m.group(2).lower()[:3]], int(m.group(1)))
    except ValueError:
        return None


def all_dates(text: str):
    out = []
    for m in _DATE_RE.finditer(text or ""):
        try:
            out.append(datetime.date(int(m.group(3)), _MONTHS[m.group(2).lower()[:3]],
                                     int(m.group(1))))
        except ValueError:
            pass
    return out


def fmt(d: datetime.date) -> str:
    return d.strftime("%d %b %Y")
