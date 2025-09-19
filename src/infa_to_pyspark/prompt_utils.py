"""Prompt utilities for reducing token usage and improving determinism."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Tuple

DEFAULT_MAX_FIELDS = 18
DEFAULT_MAX_TRANSFORMS = 200
DEFAULT_MAX_CONNECTORS = 400


def _trim_fields(fields: Iterable[dict] | None, limit: int) -> List[dict]:
    out: List[dict] = []
    if not fields:
        return out
    for idx, field in enumerate(fields):
        if limit and idx >= limit:
            remaining = len(fields) - idx if hasattr(fields, '__len__') else None
            marker = {
                "name": "__omitted__",
                "note": f"{remaining if remaining is not None else 'additional'} fields truncated for prompt compactness",
            }
            out.append(marker)
            break
        if isinstance(field, dict):
            entry = {k: field[k] for k in ["name", "datatype", "precision", "scale", "expression"] if k in field and field[k]}
            if entry:
                out.append(entry)
    return out


def compress_ast(ast_json: str, max_fields: int = DEFAULT_MAX_FIELDS, max_transforms: int = DEFAULT_MAX_TRANSFORMS, max_connectors: int = DEFAULT_MAX_CONNECTORS) -> str:
    """Return a trimmed AST representation to stabilize prompts."""
    try:
        ast = json.loads(ast_json) if ast_json else {}
    except json.JSONDecodeError:
        return ast_json or ""

    def _clone_sources(sources: Iterable[dict] | None) -> List[dict]:
        items: List[dict] = []
        for src in sources or []:
            if not isinstance(src, dict):
                continue
            entry = {"name": src.get("name"), "type": src.get("type")}
            if src.get("fields"):
                entry["fields"] = _trim_fields(src.get("fields"), max_fields)
            items.append(entry)
        return items

    def _clone_targets(targets: Iterable[dict] | None) -> List[dict]:
        items: List[dict] = []
        for tgt in targets or []:
            if not isinstance(tgt, dict):
                continue
            entry = {"name": tgt.get("name"), "type": tgt.get("type")}
            if tgt.get("fields"):
                entry["fields"] = _trim_fields(tgt.get("fields"), max_fields)
            items.append(entry)
        return items

    def _clone_transformations(transforms: Iterable[dict] | None) -> List[dict]:
        items: List[dict] = []
        if not transforms:
            return items
        transforms_list = list(transforms)
        total = len(transforms_list)
        for idx, tr in enumerate(transforms_list):
            if max_transforms and idx >= max_transforms:
                items.append({"name": "__omitted__", "note": f"{total - idx} transformations truncated"})
                break
            if not isinstance(tr, dict):
                continue
            entry = {"name": tr.get("name"), "type": tr.get("type")}
            attrs = tr.get("table_attributes")
            if isinstance(attrs, dict):
                compact_attrs = {k: attrs[k] for k in sorted(attrs.keys())[:12] if attrs[k]}
                if compact_attrs:
                    entry["table_attributes"] = compact_attrs
            if tr.get("fields"):
                entry["fields"] = _trim_fields(tr.get("fields"), max_fields)
            items.append(entry)
        return items

    compact = {
        "mapping_name": ast.get("mapping_name"),
        "sources": _clone_sources(ast.get("sources")),
        "targets": _clone_targets(ast.get("targets")),
        "transformations": _clone_transformations(ast.get("transformations")),
    }

    connectors = list(ast.get("connectors") or [])
    trimmed_connectors: List[dict] = []
    total_conn = len(connectors)
    for idx, conn in enumerate(connectors):
        if max_connectors and idx >= max_connectors:
            trimmed_connectors.append({"note": f"{total_conn - idx} connectors truncated"})
            break
        if isinstance(conn, dict):
            trimmed_connectors.append({
                "from": f"{conn.get('from_instance')}::{conn.get('from_field')}",
                "to": f"{conn.get('to_instance')}::{conn.get('to_field')}",
            })
    if trimmed_connectors:
        compact["connectors"] = trimmed_connectors

    return json.dumps(compact, indent=2)


def assemble_context(sections: List[Tuple[str, str]], max_chars: int = 7000) -> str:
    """Build prompt context, dropping lowest-priority sections until under budget."""
    cleaned: List[Tuple[str, str]] = []
    for label, value in sections:
        if not value:
            continue
        text = value.strip()
        if not text:
            continue
        cleaned.append((label, text))

    if not cleaned:
        return ""

    def _join(parts: List[Tuple[str, str]]) -> str:
        return "\n\n".join(f"=== {label} ===\n{body}" for label, body in parts)

    parts = cleaned[:]
    output = _join(parts)
    if len(output) <= max_chars:
        return output

    for _ in range(len(parts) - 1, -1, -1):
        parts.pop()
        if not parts:
            break
        output = _join(parts)
        if len(output) <= max_chars:
            return output

    return _join(parts)


def shorten_text(text: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    sliced = text[: max_chars - 3].rstrip()
    return f"{sliced}..."
