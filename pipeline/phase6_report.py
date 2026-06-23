"""Phase 6: Generate aggregate report.

Computes folder-level aggregates, writes preflight_report.md with
all view outputs and inferred folder taxonomy.
Also runs language detection aggregation for bilingual corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

from pipeline.config import Config
from pipeline.helpers import now_iso
from pipeline.logging import PhaseLogger


def _fmt_table(rows: list[dict], title: str) -> str:
    """Format rows as a markdown table."""
    if not rows:
        return f"### {title}\n\n*(no data)*\n"

    headers = list(rows[0].keys())
    lines = [f"### {title}", ""]
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h) or "") for h in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def _fmt_list(items: list[str], title: str) -> str:
    if not items:
        return f"### {title}\n\n*(none)*\n"
    lines = [f"### {title}", ""]
    for item in items:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _build_folder_tree(db) -> str:
    """Build an indented folder tree with inferred labels using a recursive CTE."""
    # Single CTE fetches the entire tree with depth and path info
    rows = list(
        db.query(
            """
            WITH RECURSIVE tree(id, name, parent_id, inferred_label, inferred_category, depth, path) AS (
                SELECT id, name, parent_id, inferred_label, inferred_category, depth,
                       CAST(name AS TEXT)
                FROM folder
                WHERE parent_id IS NULL OR depth = 0

                UNION ALL

                SELECT f.id, f.name, f.parent_id, f.inferred_label, f.inferred_category, f.depth,
                       CAST(t.path || '/' || f.name AS TEXT)
                FROM folder f JOIN tree t ON f.parent_id = t.id
            )
            SELECT * FROM tree ORDER BY path
            """
        )
    )

    # Build parent->children index in Python
    children_of: dict[int, list[dict]] = {}
    by_id: dict[int, dict] = {}
    roots: list[dict] = []
    for r in rows:
        r["_children"] = []
        by_id[r["id"]] = r
        pid = r.get("parent_id")
        if pid is None or pid not in by_id:
            roots.append(r)
        else:
            children_of.setdefault(pid, []).append(r)

    def _wire_children(node):
        fid = node["id"]
        for child in children_of.get(fid, []):
            node["_children"].append(child)
            _wire_children(child)

    for root in roots:
        _wire_children(root)

    def _render(node, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        label = node.get("inferred_label", "")
        cat = node.get("inferred_category", "")
        tag = f" [{cat}] {label}" if label else f" ({cat})" if cat else ""
        lines = [f"{prefix}- {node['name']}{tag}"]
        for child in node["_children"]:
            lines.extend(_render(child, indent + 1))
        return lines

    lines = ["### Folder Taxonomy\n"]
    for root in roots:
        lines.extend(_render(root))
    lines.append("")
    return "\n".join(lines)


def _build_language_summary(db) -> str:
    """Summarize detected languages from triage_json."""
    rows = list(
        db.query("""
        SELECT triage_json FROM file
        WHERE triage_json IS NOT NULL AND triage_status='done'
        AND is_dup_primary=1 AND excluded=0
    """)
    )

    lang_counts: dict[str, int] = {}
    total_with_lang = 0
    for row in rows:
        try:
            tj = json.loads(row["triage_json"])
            lang = tj.get("detected_language")
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                total_with_lang += 1
        except (json.JSONDecodeError, TypeError):
            pass

    if not lang_counts:
        return "### Language Distribution\n\n*Language detection not available (no text-bearing files triaged or lingua not installed)*\n"

    lines = ["### Language Distribution\n", "| Language | Files |"]
    lines.append("| --- | --- |")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = f"{count / total_with_lang * 100:.1f}%"
        lines.append(f"| {lang} | {count} ({pct}) |")
    lines.append("")
    return "\n".join(lines)


def run_phase6(
    db,
    config: Config,
    plog: PhaseLogger,
    output_path: str | None = None,
) -> dict:
    """Generate aggregates and markdown report."""
    # Update folder-level aggregates in a single pass
    plog.info("Computing folder-level aggregates...")

    # Single query: file stats per folder
    stats_by_folder = {
        row["folder_id"]: row
        for row in db.query(
            """
            SELECT folder_id,
                   COUNT(*) AS file_count,
                   COALESCE(SUM(size_bytes), 0) AS total_bytes,
                   SUM(CASE WHEN category IN ('document','spreadsheet','presentation','pdf-text','pdf-scanned')
                            THEN 1 ELSE 0 END) AS document_count
            FROM file WHERE excluded=0 GROUP BY folder_id
            """
        )
    }

    # Single query: dominant format per folder via window function
    dom_by_folder = {
        row["folder_id"]: row["format_name"]
        for row in db.query(
            """
            SELECT folder_id, format_name FROM (
                SELECT folder_id, format_name,
                       ROW_NUMBER() OVER (PARTITION BY folder_id ORDER BY COUNT(*) DESC) AS rn
                FROM file WHERE excluded=0 AND format_name IS NOT NULL
                GROUP BY folder_id, format_name
            ) WHERE rn = 1
            """
        )
    }

    folders = list(db.query("SELECT * FROM folder ORDER BY depth"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Aggregating...", total=len(folders))
        for folder in folders:
            fid = folder["id"]
            stats = stats_by_folder.get(fid, {})
            dom = dom_by_folder.get(fid)

            db["folder"].update(
                fid,
                {
                    "file_count": stats.get("file_count", 0),
                    "total_bytes": stats.get("total_bytes", 0),
                    "document_count": stats.get("document_count", 0),
                    "dominant_format": dom,
                },
            )
            progress.update(task, advance=1)

    # Build report
    sections = ["# RAG Pre-Flight Report\n"]
    sections.append(f"Generated: {now_iso()}\n")
    sections.append(f"Corpus root: `{config.corpus_root_path}`\n")

    # Total files and folders
    totals = next(
        iter(db.query("""
        SELECT COUNT(*) AS total_files,
               SUM(size_bytes) AS total_bytes,
               SUM(CASE WHEN is_dup_primary=1 AND excluded=0 THEN 1 ELSE 0 END) AS unique_files,
               SUM(CASE WHEN excluded=1 THEN 1 ELSE 0 END) AS excluded_files
        FROM file
    """)),
        {},
    )

    folder_count = next(iter(db.query("SELECT COUNT(*) AS total_folders FROM folder")), {})

    sections.append("## Corpus Overview\n")
    sections.append(f"- **Total files**: {totals.get('total_files', 0)}")
    sections.append(f"- **Total folders**: {folder_count.get('total_folders', 0)}")
    sections.append(
        f"- **Total size**: {(totals.get('total_bytes', 0) or 0) / (1024 * 1024):.1f} MB"
    )
    sections.append(f"- **Unique files** (after dedup): {totals.get('unique_files', 0)}")
    sections.append(f"- **Excluded files**: {totals.get('excluded_files', 0)}")
    sections.append("")

    # Format histogram
    formats = list(db.query("SELECT * FROM v_format_histogram LIMIT 30"))
    sections.append(_fmt_table(formats, "Top 30 Formats (by file count)"))

    # Category summary
    cats = list(db.query("SELECT * FROM v_category_summary"))
    sections.append(_fmt_table(cats, "Category Breakdown"))

    # OCR workload
    ocr = list(db.query("SELECT * FROM v_ocr_workload"))
    if ocr:
        r = ocr[0]
        sections.append("### OCR Workload\n")
        sections.append(f"- Scanned PDFs: {r['scanned_pdfs']}")
        sections.append(f"- Total pages to OCR: {r['total_pages']}\n")

    # Transcription workload
    trans = list(db.query("SELECT * FROM v_transcription_workload"))
    if trans:
        sections.append(_fmt_table(trans, "Transcription Workload"))

    # Duplicate summary
    dups = list(db.query("SELECT * FROM v_duplicate_summary LIMIT 20"))
    if dups:
        sections.append(_fmt_table(dups, "Top 20 Duplicate Groups"))

    # Problem files
    problems = list(db.query("SELECT * FROM v_problem_files LIMIT 50"))
    if problems:
        sections.append(_fmt_table(problems, "Problem Files (top 50)"))

    # Unknown formats
    unknowns = list(db.query("SELECT * FROM v_unknown_formats"))
    if unknowns:
        sections.append(_fmt_table(unknowns, "Unknown Formats (need policy entries)"))

    # Language distribution
    sections.append(_build_language_summary(db))

    # Folder taxonomy
    sections.append(_build_folder_tree(db))

    # Failure summary
    failures = list(
        db.query("""
        SELECT phase, error_class, COUNT(*) AS n FROM failure
        GROUP BY phase, error_class ORDER BY n DESC
    """)
    )
    if failures:
        sections.append(_fmt_table(failures, "Failure Summary"))

    # Write report
    report_path = (
        Path(output_path) if output_path else config.corpus_root_path / "preflight_report.md"
    )
    report_path.write_text("\n".join(sections), encoding="utf-8")
    plog.info(f"Report written to {report_path}")

    # Mark preflight complete
    db["config"].insert({"key": "preflight_completed_at", "value": now_iso()}, replace=True)

    return {"files_processed": totals.get("total_files", 0)}
