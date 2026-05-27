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


def _parse_one(path: Path) -> Skill | None:
    text = path.read_text(encoding="utf-8")
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


@lru_cache(maxsize=1)
def _all_skills() -> List[Skill]:
    if not _LIBRARY.exists():
        return []
    skills: list[Skill] = []
    for md in sorted(_LIBRARY.glob("*.md")):
        if md.name.lower() == "readme.md":
            continue
        skill = _parse_one(md)
        if skill is not None:
            skills.append(skill)
    return skills


def load_manifest() -> str:
    skills = _all_skills()
    if not skills:
        return "(no skills available)"
    return "\n".join(f"- {s.id}: {s.description}" for s in skills)


def load_skills(ids: Iterable[str]) -> str:
    ids_list = list(ids)
    skills = _all_skills()
    by_id = {s.id: s for s in skills}

    selected: list[Skill] = []
    if "ALL" in ids_list:
        selected = skills
    else:
        for sid in ids_list:
            if sid in by_id:
                selected.append(by_id[sid])
    return "\n\n".join(s.body for s in selected) if selected else ""
