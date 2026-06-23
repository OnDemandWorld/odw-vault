"""Phase 2: Format identification using Siegfried (PRONOM signatures).

Runs `sf -json -multi N <corpus_root>`, parses output, joins to `file` by
absolute path. Assigns category and extract_strategy from format_policy table.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

from pipeline.config import Config
from pipeline.helpers import record_failure
from pipeline.logging import PhaseLogger

# Extension fallback for files where Siegfried returns UNKNOWN or no match
# Maps extension -> (pronom_id, category, extract_strategy)
EXT_FALLBACK = {
    ".pak": ("UNKNOWN-pak", "data", "filename-only"),
    ".qm": ("UNKNOWN-qm", "data", "filename-only"),
    ".drawio": ("UNKNOWN-drawio", "data", "filename-only"),
    ".bak": ("UNKNOWN-bak", "data", "filename-only"),
    ".tmp": ("UNKNOWN-tmp", "data", "filename-only"),
    ".dat": ("UNKNOWN-dat", "data", "filename-only"),
    ".log": ("UNKNOWN-log", "document", "tika"),
    ".cfg": ("UNKNOWN-cfg", "data", "filename-only"),
    ".ini": ("UNKNOWN-ini", "data", "filename-only"),
    ".bin": ("UNKNOWN-bin", "executable", "skip"),
    ".db": ("UNKNOWN-db", "data", "filename-only"),
    ".sqlite": ("UNKNOWN-sqlite", "data", "filename-only"),
}


def _apply_extension_fallback(filepath: str) -> tuple:
    """Try to categorize by file extension when Siegfried fails."""
    ext = Path(filepath).suffix.lower()
    if ext in EXT_FALLBACK:
        return EXT_FALLBACK[ext]
    return (None, None, None)


def run_phase2(
    db,
    config: Config,
    plog: PhaseLogger,
    reidentify: bool = False,
) -> dict:
    """Run Siegfried format identification over the corpus."""
    sf_path = config.identify.siegfried_path
    n_workers = config.identify.siegfried_workers
    root = config.corpus_root_path

    # Resolve siegfried path: check PATH first, then project root
    sf_path = config.identify.siegfried_path
    if not shutil.which(sf_path):
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / sf_path
        if candidate.is_file():
            sf_path = str(candidate)
        else:
            plog.error(f"Siegfried binary '{sf_path}' not found on PATH or in project root")
            raise RuntimeError(f"Siegfried not found: {sf_path}")

    # Find files that need identification
    if reidentify:
        files_to_id = [
            r["path"] for r in db["file"].rows_where("hash_status='done' AND excluded=0")
        ]
    else:
        files_to_id = [
            r["path"]
            for r in db["file"].rows_where(
                "identify_status='pending' AND hash_status='done' AND excluded=0"
            )
        ]

    if not files_to_id:
        plog.info("No files need identification.")
        return {"files_processed": 0, "files_failed": 0}

    plog.info(f"Running Siegfried on {len(files_to_id)} files...")

    try:
        result = subprocess.run(
            [sf_path, "-json", "-multi", str(n_workers), str(root)],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0 and not result.stdout:
            plog.error(f"Siegfried returned non-zero: {result.stderr}")
            raise RuntimeError(f"Siegfried failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        plog.error("Siegfried timed out after 1 hour")
        raise RuntimeError("Siegfried timeout") from None

    try:
        sf_data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        plog.error(f"Failed to parse Siegfried JSON: {e}")
        raise RuntimeError(f"Siegfried JSON parse error: {e}") from None

    # Build path -> identification map
    sf_results = {}
    for file_entry in sf_data.get("files", []):
        filepath = file_entry.get("filename", "")
        sf_results[filepath] = file_entry

    identified = 0
    unknown = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Applying format IDs...", total=len(files_to_id))

        for filepath in files_to_id:
            sf_entry = sf_results.get(filepath)

            if sf_entry and sf_entry.get("matches"):
                ident = sf_entry["matches"][0]
                puid = ident.get("id", "")
                mime = ident.get("mime", "")
                format_name = ident.get("format", "")
                format_version = ident.get("version", "")
                id_warning = sf_entry.get("warning", "")

                # Siegfried returns "UNKNOWN" or empty string when no signature matched
                if not puid or puid == "UNKNOWN":
                    ext_puid, ext_cat, ext_strategy = _apply_extension_fallback(filepath)
                    if ext_cat:
                        puid = ext_puid or "UNKNOWN"
                        mime = ""
                        category = ext_cat
                        extract_strategy = ext_strategy
                        format_name = f"Unknown ({ext_puid})" if ext_puid else "Unknown"
                        format_version = ""
                    else:
                        category = "unknown"
                        extract_strategy = "manual"
                        unknown += 1
                else:
                    policy = next(db["format_policy"].rows_where("pronom_id = ?", [puid]), None)
                    if policy:
                        category = policy["category"]
                        extract_strategy = policy["extract_strategy"]
                    else:
                        # PRONOM ID not in policy — try extension fallback
                        ext_puid, ext_cat, ext_strategy = _apply_extension_fallback(filepath)
                        if ext_cat:
                            category = ext_cat
                            extract_strategy = ext_strategy
                        else:
                            category = "unknown"
                            extract_strategy = "manual"
                            unknown += 1
                            record_failure(
                                db,
                                phase="identify",
                                tool="siegfried",
                                error_class="unsupported_format",
                                error_message=f"Unknown PRONOM ID: {puid} ({format_name}) for {filepath}",
                            )

                sf_json = json.dumps(sf_entry, ensure_ascii=False)

                db.execute(
                    """
                    UPDATE file SET pronom_id=?, mime_type=?, format_name=?,
                        format_version=?, siegfried_json=?, id_warning=?,
                        category=?, extract_strategy=?, identify_status='done'
                    WHERE path=?
                """,
                    [
                        puid or None,
                        mime or None,
                        format_name or None,
                        format_version or None,
                        sf_json,
                        id_warning or None,
                        category,
                        extract_strategy,
                        filepath,
                    ],
                )
                identified += 1
            else:
                # Siegfried returned no match — try extension fallback
                ext_puid, ext_cat, ext_strategy = _apply_extension_fallback(filepath)
                if ext_cat:
                    db.execute(
                        """
                        UPDATE file SET pronom_id=?, category=?, extract_strategy=?,
                            format_name=?, identify_status='done' WHERE path=?
                    """,
                        [ext_puid, ext_cat, ext_strategy, f"Unknown ({ext_puid})", filepath],
                    )
                    unknown += 1
                else:
                    db.execute(
                        """
                        UPDATE file SET category='unknown', extract_strategy='manual',
                            identify_status='done' WHERE path=?
                    """,
                        [filepath],
                    )
                    unknown += 1

            progress.update(task, advance=1)

    plog.info(
        f"Identification complete: {identified} identified, {unknown} unknown",
        identified=identified,
        unknown=unknown,
    )
    return {"files_processed": identified, "files_failed": 0}
