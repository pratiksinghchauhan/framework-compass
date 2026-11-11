"""Turn MITRE ATLAS v6 YAML into retrieval units, one per object."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import yaml

INTERNAL_LINK = re.compile(r"\[([^\]]+)\]\(/[a-z-]+/([A-Za-z0-9.\-]+)\)")
EXTERNAL_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|mailto:)[^)]*\)")

URL_PATHS = {
    "tactic": "tactics",
    "technique": "techniques",
    "mitigation": "mitigations",
    "case-study": "studies",
}


def _markdown_to_text(text) -> str:
    text = str(text or "")
    text = INTERNAL_LINK.sub(lambda m: m[1] if m[2] in m[1] else f"{m[1]} ({m[2]})", text)
    return EXTERNAL_LINK.sub(r"\1", text).strip()


def _paragraphs(text) -> list[str]:
    parts = re.split(r"\n\s*\n", _markdown_to_text(text))
    return [re.sub(r"\s*\n\s*", " ", p).strip() for p in parts if p.strip()]


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [] if value in (None, "", "[]") else [str(value)]


def load_units(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    release = str(data["collection"]["version"])
    objects = {}
    for kind in ("tactics", "techniques", "mitigations", "case-studies", "entities", "software"):
        objects.update(data.get(kind) or {})

    def label(obj_id: str) -> str:
        obj = objects.get(obj_id)
        return f"{obj_id} {obj['name']}" if obj else obj_id

    def label_list(ids) -> str:
        return "; ".join(label(i) for i in ids)

    out_links = defaultdict(lambda: defaultdict(list))  # source -> type -> [rel]
    in_links = defaultdict(lambda: defaultdict(list))   # target -> type -> [rel]
    for source, by_type in (data.get("relationships") or {}).items():
        for rel_type, rels in by_type.items():
            for rel in rels:
                out_links[source][rel_type].append(rel)
                in_links[rel["target"]][rel_type].append(rel)

    def targets(obj_id, rel_type):
        return [r["target"] for r in out_links[obj_id][rel_type]]

    def sources(obj_id, rel_type):
        return [r["source"] for r in in_links[obj_id][rel_type]]

    units = []

    def add(kind, section, obj, paras, extra_path=(), meta=None):
        obj_id = obj["id"]
        meta = dict(meta or {})
        meta["atlas_release"] = release
        if kind in URL_PATHS:
            meta["url"] = f"https://atlas.mitre.org/{URL_PATHS[kind]}/{obj_id}"
        units.append({
            "kind": kind,
            "path": [section, label(obj_id), *extra_path],
            "paras": [p for p in paras if p],
            "primary_id": obj_id,
            "meta": meta,
        })

    for obj in (data.get("tactics") or {}).values():
        techniques = sources(obj["id"], "achieves")
        add("tactic", "Tactics", obj, [
            *_paragraphs(obj.get("description")),
            f"Techniques for this tactic: {label_list(techniques)}." if techniques else "",
        ])

    for obj in (data.get("techniques") or {}).values():
        obj_id = obj["id"]
        parents = targets(obj_id, "specializes")
        tactics = targets(obj_id, "achieves") or [t for p in parents for t in targets(p, "achieves")]
        mitigations = sources(obj_id, "mitigates")
        subtechniques = sources(obj_id, "specializes")
        facts = [f"Tactics: {label_list(tactics)}." if tactics else ""]
        if parents:
            facts.append(f"Sub-technique of {label_list(parents)}.")
        if obj.get("maturity"):
            facts.append(f"Maturity: {obj['maturity']}.")
        platforms = _as_list(obj.get("platforms"))
        if platforms:
            facts.append(f"Platforms: {', '.join(platforms)}.")
        add("technique", "Techniques", obj, [
            " ".join(f for f in facts if f),
            *_paragraphs(obj.get("description")),
            f"Sub-techniques: {label_list(subtechniques)}." if subtechniques else "",
            f"Mitigations: {label_list(mitigations)}." if mitigations else "",
        ], meta={"tactics": tactics})

    for obj in (data.get("mitigations") or {}).values():
        obj_id = obj["id"]
        rels = out_links[obj_id]["mitigates"]
        facts = []
        if _as_list(obj.get("categories")):
            facts.append(f"Category: {', '.join(_as_list(obj['categories']))}.")
        if _as_list(obj.get("lifecycle-phases")):
            facts.append(f"Lifecycle phases: {', '.join(_as_list(obj['lifecycle-phases']))}.")
        add("mitigation", "Mitigations", obj, [
            " ".join(facts),
            *_paragraphs(obj.get("description")),
            f"Mitigates: {label_list(r['target'] for r in rels)}." if rels else "",
        ])
        for rel in rels:
            if rel.get("description"):
                add("mitigation-use", "Mitigations", obj,
                    _paragraphs(rel["description"]),
                    extra_path=[f"Applied to {label(rel['target'])}"],
                    meta={"technique": rel["target"]})

    for obj in (data.get("case-studies") or {}).values():
        obj_id = obj["id"]
        facts = [f"{key.title()}: {obj[key]}." for key in ("type", "actor", "target") if obj.get(key)]
        if obj.get("start-date"):
            facts.append(f"Date: {obj['start-date']}.")
        people = targets(obj_id, "attributed-to") + targets(obj_id, "reported-by")
        software = targets(obj_id, "uses")
        add("case-study", "Case studies", obj, [
            " ".join(facts),
            *_paragraphs(obj.get("description")),
            f"Attributed to or reported by: {label_list(people)}." if people else "",
            f"Software used: {label_list(software)}." if software else "",
        ])
        steps = []
        for n, rel in enumerate(out_links[obj_id]["employs"], start=1):
            text = " ".join(_paragraphs(rel.get("description")))
            steps.append(f"Step {n}, {label(rel['target'])}: {text}".rstrip(": "))
        if steps:
            add("case-study-procedure", "Case studies", obj, steps, extra_path=["Procedure"],
                meta={"techniques": [r["target"] for r in out_links[obj_id]["employs"]]})

    for section, kind, key in (("Entities", "entity", "entities"), ("Software", "software", "software")):
        for obj in (data.get(key) or {}).values():
            kind_field = obj.get("entity-type") or obj.get("software-type")
            linked = targets(obj["id"], "implements") + targets(obj["id"], "uses")
            add(kind, section, obj, [
                f"Type: {kind_field}." if kind_field else "",
                *_paragraphs(obj.get("description")),
                f"Related techniques and software: {label_list(linked)}." if linked else "",
            ])

    return units
