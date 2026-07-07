"""
Knowledge base loader.

Loads the JSON files produced by preprocess.py into lightweight, read-only
Python structures. No PDF parsing here - the agent only ever sees the
preprocessed knowledge. Everything is deterministic and offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


KB_FILES = ("manifest", "reports", "chunks", "verdicts", "facts", "entities")


@dataclass(frozen=True)
class Source:
    """A citation back to the reports - what the frontend renders as provenance."""
    report_id: str
    page: Optional[int] = None
    section: Optional[str] = None

    def label(self) -> str:
        parts = [self.report_id]
        if self.page:
            parts.append("p%d" % self.page)
        return " ".join(parts)


class KnowledgeBase:
    """In-memory view over the preprocessed knowledge files."""

    def __init__(self, data: Dict[str, Any]):
        self.manifest: Dict[str, Any] = data.get("manifest", {})
        self.reports: List[Dict] = data.get("reports", [])
        self.chunks: List[Dict] = data.get("chunks", [])
        self.verdicts: List[Dict] = data.get("verdicts", [])
        self.facts: List[Dict] = data.get("facts", [])
        self.entities: List[Dict] = data.get("entities", [])

        # Fast lookups (insertion-ordered => deterministic iteration).
        self.report_by_id: Dict[str, Dict] = {r["report_id"]: r for r in self.reports}
        self.entity_by_name: Dict[str, Dict] = {e["name"]: e for e in self.entities}

        # Module -> report ids
        self.reports_by_module: Dict[str, List[str]] = {}
        for r in self.reports:
            self.reports_by_module.setdefault(r["module"], []).append(r["report_id"])

    # -- factory ----------------------------------------------------------- #

    @classmethod
    def load(cls, knowledge_dir: str) -> "KnowledgeBase":
        data: Dict[str, Any] = {}
        for name in KB_FILES:
            path = os.path.join(knowledge_dir, name + ".json")
            if not os.path.exists(path):
                if name == "manifest":
                    data[name] = {}
                    continue
                raise FileNotFoundError(
                    "Missing knowledge file: %s\n"
                    "Run:  python preprocess.py --out %s" % (path, knowledge_dir)
                )
            with open(path, "r", encoding="utf-8") as fh:
                data[name] = json.load(fh)
        return cls(data)

    # -- convenience ------------------------------------------------------- #

    def module_label(self, module: str) -> str:
        for r in self.reports:
            if r["module"] == module:
                return r.get("module_label", module)
        return module

    def report_label(self, report_id: str) -> str:
        r = self.report_by_id.get(report_id)
        if not r:
            return report_id
        bits = [report_id]
        if r.get("period"):
            bits.append("(" + r["period"] + ")")
        return " ".join(bits)

    def stats(self) -> Dict[str, int]:
        return {
            "reports": len(self.reports),
            "chunks": len(self.chunks),
            "verdicts": len(self.verdicts),
            "facts": len(self.facts),
            "entities": len(self.entities),
        }
