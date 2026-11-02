"""Configurable grammar profile for legal document ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ParserProfile:
    """Grammar profile for a family of legal documents."""

    name: str
    default_namespace: str = "decision"
    section_markers: dict[str, str] = field(default_factory=lambda: {"QUY CHẾ": "regulation"})
    chapter_tokens: tuple[str, ...] = ("Chương", "CHƯƠNG")
    article_tokens: tuple[str, ...] = ("Điều",)
    clause_punctuation: tuple[str, ...] = (".", ":", ")")
    point_styles: tuple[str, ...] = ("a)", "a.", "(a)", "A)")
    subpoint_bullets: tuple[str, ...] = ("-", "–", "+", "*")
    crossref_terms: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "dieu": ("Điều", "điều"),
            "khoan": ("Khoản", "khoản"),
            "diem": ("Điểm", "điểm"),
            "muc": ("Mục", "mục"),
        }
    )
    crossref_targets: dict[str, str] = field(
        default_factory=lambda: {
            "dieu": r"\d+[A-Za-z]?(?![^\W_])",
            "khoan": r"\d+[A-Za-z]?(?![^\W_])",
            "diem": r"[a-z](?![^\W_])",
            "muc": r"(?:\d+|[IVXLCDMivxlcdm]+|[a-z])(?![^\W_])",
        }
    )

    def chapter_re(self) -> re.Pattern[str]:
        tokens = "|".join(re.escape(token) for token in self.chapter_tokens)
        return re.compile(rf"^({tokens})\s+([IVXLCDM]+)\s*(?:[.:]\s*(.*))?$")

    def article_re(self) -> re.Pattern[str]:
        tokens = "|".join(re.escape(token) for token in self.article_tokens)
        return re.compile(rf"^({tokens})\s+(\d+[a-zA-Z]*)(?:[.:])\s*(.*)$")

    def clause_re(self) -> re.Pattern[str]:
        punctuation = "".join(re.escape(ch) for ch in self.clause_punctuation)
        return re.compile(rf"^(\d+)(?:[{punctuation}])\s+(.*)$")

    def point_re(self) -> re.Pattern[str]:
        patterns = []
        for style in self.point_styles:
            style = style.strip()
            if len(style) == 2 and style[0].isalpha() and style[1] == ")":
                patterns.append(rf"([a-zA-Z])\)")
            elif len(style) == 2 and style[0].isalpha() and style[1] == ".":
                patterns.append(rf"([a-zA-Z])\.")
            elif len(style) == 3 and style[0] == "(" and style[2] == ")":
                patterns.append(rf"\(([a-zA-Z])\)")
        combined = "|".join(f"(?:{pattern})" for pattern in patterns)
        return re.compile(rf"^(?:{combined})\s+(.*)$")

    def subpoint_re(self) -> re.Pattern[str]:
        bullets = "".join(re.escape(ch) for ch in self.subpoint_bullets)
        return re.compile(rf"^[{bullets}]\s+(.*)$")

    def table_re(self) -> re.Pattern[str]:
        return re.compile(r"^\|.*\|$")

    def table_separator_re(self) -> re.Pattern[str]:
        return re.compile(r"^\|[-:\s|]+\|$")

    def structural_kinds(self) -> tuple[str, ...]:
        return ("chapter", "article", "clause", "point", "subpoint")

    def looks_structural(self, line: str) -> str | None:
        if self.chapter_re().match(line):
            return "chapter"
        if self.article_re().match(line):
            return "article"
        if self.clause_re().match(line):
            return "clause"
        if self.point_re().match(line):
            return "point"
        if self.subpoint_re().match(line):
            return "subpoint"
        return None

    def crossref_res(self) -> dict[str, re.Pattern[str]]:
        return {
            kind: re.compile(
                r"(?<!\w)("
                + "|".join(re.escape(term) for term in terms)
                + rf")\s+(?P<target>{self.crossref_targets[kind]})"
            )
            for kind, terms in self.crossref_terms.items()
        }


def vietnamese_regulation_profile() -> ParserProfile:
    return ParserProfile(name="vietnamese_regulation")
