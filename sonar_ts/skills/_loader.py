"""Skill loader."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

_LIBRARY = Path(__file__).resolve().parent / "library"


@dataclass(frozen=True)
class Skill:
    id: str
    description: str
    body: str


def parse_skill_text(text: str) -> Skill | None:
    """Parse a skill document (frontmatter `id` / `description` + body)."""
    text = text.lstrip("﻿").lstrip()
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) != 3:
        return None
    header, body = parts[1], parts[2].lstrip("\n")
    meta: dict[str, str] = {}
    for line in header.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    skill_id = meta.get("id")
    description = meta.get("description", "").strip()
    if not skill_id:
        return None
    return Skill(id=skill_id, description=description, body=body.strip())


def _parse_one(path: Path) -> Skill | None:
    return parse_skill_text(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _all_skills() -> List[Skill]:
    return _read_dir(_LIBRARY)


def _read_dir(directory: Path) -> List[Skill]:
    """Parse every `*.md` skill in `directory` (sorted, README ignored)."""
    if not directory.exists():
        return []
    skills: list[Skill] = []
    for md in sorted(directory.glob("*.md")):
        if md.name.lower() == "readme.md":
            continue
        skill = _parse_one(md)
        if skill is not None:
            skills.append(skill)
    return skills


def _manifest_of(skills: List[Skill]) -> str:
    if not skills:
        return "(no skills available)"
    return "\n".join(f"- {s.id}: {s.description}" for s in skills)


def _bodies_of(skills: List[Skill], ids: Iterable[str]) -> str:
    ids_list = list(ids)
    by_id = {s.id: s for s in skills}
    if "ALL" in ids_list:
        selected = list(skills)
    else:
        selected = [by_id[sid] for sid in ids_list if sid in by_id]
    return "\n\n".join(s.body for s in selected) if selected else ""


class SkillLibrary:
    """A skill library living in an arbitrary directory.

    Unlike the module-level helpers this class is *not* cached: the directory
    is re-read on every call, so a library that grows during a run stays
    current.
    """

    def __init__(self, directory: Path | str):
        self.dir = Path(directory)

    def skills(self) -> List[Skill]:
        return _read_dir(self.dir)

    def ids(self) -> List[str]:
        return [s.id for s in self.skills()]

    def get(self, skill_id: str) -> Skill | None:
        for s in self.skills():
            if s.id == skill_id:
                return s
        return None

    def manifest(self) -> str:
        return _manifest_of(self.skills())

    def load(self, ids: Iterable[str]) -> str:
        return _bodies_of(self.skills(), ids)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SkillLibrary({str(self.dir)!r}, n={len(self.skills())})"


def load_manifest() -> str:
    return _manifest_of(_all_skills())


def load_skills(ids: Iterable[str]) -> str:
    return _bodies_of(_all_skills(), ids)
