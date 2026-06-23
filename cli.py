"""CLI entry point for rag-preflight. All subcommands defined here."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from pipeline.config import DEFAULT_CONFIG_TOML, Config, load_app_config, load_config
from pipeline.db import cleanup_stale_runs, is_db_initialized, migrate, open_db

# Project root is the directory containing cli.py
PROJECT_ROOT = Path(__file__).resolve().parent


def get_config(ctx: click.Context) -> Config:
    """Load config from config.toml in project root."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        click.echo(f"Error: config.toml not found at {config_path}. Run 'init' first.", err=True)
        ctx.exit(2)
    try:
        return load_config(config_path)
    except Exception as e:
        click.echo(f"Error loading config.toml: {e}", err=True)
        ctx.exit(2)


def get_db_path(ctx: click.Context) -> Path:
    return PROJECT_ROOT / "corpus.db"


@click.group()
def main():
    """Local RAG Pre-Flight Pipeline — inventory, identify, and triage document corpora."""
    pass


@main.command()
@click.option("--root", "corpus_root", required=True, help="Absolute path to corpus root")
@click.option(
    "--cache", "cache_root", default=None, help="Cache directory (default: <root>/.rag-cache)"
)
@click.option("--force", is_flag=True, help="Overwrite existing corpus.db")
def init(corpus_root: str, cache_root: str | None, force: bool) -> None:
    """Initialize the database and configuration."""
    root = Path(corpus_root).resolve()
    if not root.is_dir():
        click.echo(f"Error: {root} does not exist or is not a directory", err=True)
        sys.exit(2)

    cache = Path(cache_root or str(root / ".rag-cache")).resolve()
    db_path = PROJECT_ROOT / "corpus.db"

    if db_path.exists() and not force:
        click.echo(f"Error: {db_path} already exists. Use --force to overwrite.", err=True)
        sys.exit(1)

    if db_path.exists() and force:
        db_path.unlink()

    # Write config.toml (preserve existing if present, only update paths)
    config_path = PROJECT_ROOT / "config.toml"
    if config_path.exists() and not force:
        # Update existing config with new paths
        import tomllib

        import tomli_w

        with open(config_path, "rb") as f:
            existing = tomllib.load(f)
        existing["paths"] = {"corpus_root": str(root), "cache_root": str(cache)}
        config_path.write_text(tomli_w.dumps(existing), encoding="utf-8")
        click.echo(f"Updated {config_path} with new paths")
    else:
        config_content = DEFAULT_CONFIG_TOML.replace(
            'corpus_root = "/path/to/corpus"', f'corpus_root = "{root}"'
        ).replace('cache_root = "/path/to/corpus/.rag-cache"', f'cache_root = "{cache}"')
        config_path.write_text(config_content, encoding="utf-8")
        click.echo(f"Created {config_path}")

    # Create cache directories
    for subdir in ["extractions", "archives", "logs", "models"]:
        (cache / subdir).mkdir(parents=True, exist_ok=True)
    click.echo(f"Created cache directory: {cache}")

    # Create DB and apply schema
    db = open_db(db_path)
    migrate(db)
    click.echo(f"Created {db_path} with full schema")

    # Seed format policy
    seed_path = PROJECT_ROOT / "seeds" / "format_policy.csv"
    if seed_path.exists():
        import csv

        with open(seed_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            db["format_policy"].insert_all(reader, replace=True)
        click.echo(f"Seeded format_policy from {seed_path}")

    # Populate DB config table so status command works
    db["config"].insert({"key": "corpus_root", "value": str(root)}, replace=True)
    db["config"].insert({"key": "cache_root", "value": str(cache)}, replace=True)
    db["config"].insert({"key": "pipeline_version", "value": "0.1.0"}, replace=True)
    click.echo("Populated DB config table")

    click.echo("Initialization complete. Run 'rag-preflight archives' to begin.")


def _run_phase(ctx: click.Context, phase_name: str, phase_fn, **kwargs):
    """Common wrapper for phase execution."""
    config = get_config(ctx)
    db_path = get_db_path(ctx)
    if not is_db_initialized(db_path):
        click.echo("Error: database not initialized. Run 'init' first.", err=True)
        ctx.exit(1)
    db = open_db(db_path)

    from pipeline.logging import PhaseLogger

    plog = PhaseLogger(phase_name, config.cache_root_path)

    # Record run
    run_id = (
        db["pipeline_run"]
        .insert(
            {
                "phase": phase_name,
                "status": "running",
            }
        )
        .last_rowid
    )

    try:
        result = phase_fn(db, config, plog, **kwargs)
        db["pipeline_run"].update(
            run_id,
            {
                "status": "done",
                "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "files_processed": result.get("files_processed") if result else None,
                "files_failed": result.get("files_failed") if result else None,
            },
        )
        return result
    except Exception as e:
        db["pipeline_run"].update(
            run_id,
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "notes": str(e),
            },
        )
        plog.error(f"Phase {phase_name} failed: {e}")
        raise


@main.command()
@click.option("--max-depth", type=int, default=None, help="Max recursion depth for nested archives")
@click.option("--dry-run", is_flag=True, help="List archives without extracting")
def archives(max_depth: int | None, dry_run: bool) -> None:
    """Phase 0: Expand archive files."""
    from pipeline.phase0_archives import run_phase0

    def wrapper(db, config, plog, **kw):
        return run_phase0(db, config, plog, max_depth=max_depth, dry_run=dry_run)

    _run_phase(click.get_current_context(), "archives", wrapper)


@main.command()
@click.option(
    "--workers", type=int, default=None, help="Parallel hash workers (default: CPU count)"
)
@click.option("--rehash", is_flag=True, help="Force re-hash of unchanged files")
def walk(workers: int | None, rehash: bool) -> None:
    """Phase 1: Walk tree and compute SHA-256."""
    from pipeline.phase1_walk import run_phase1

    def wrapper(db, config, plog, **kw):
        return run_phase1(db, config, plog, workers=workers, rehash=rehash)

    _run_phase(click.get_current_context(), "walk", wrapper)


@main.command()
@click.option("--reidentify", is_flag=True, help="Force re-identification of all files")
def identify(reidentify: bool) -> None:
    """Phase 2: Format identification via Siegfried."""
    from pipeline.phase2_identify import run_phase2

    def wrapper(db, config, plog, **kw):
        return run_phase2(db, config, plog, reidentify=reidentify)

    _run_phase(click.get_current_context(), "identify", wrapper)


@main.command()
@click.option("--workers", type=int, default=None, help="Parallel triage workers")
@click.option("--categories", default=None, help="Comma-separated categories to triage")
def triage(workers: int | None, categories: str | None) -> None:
    """Phase 3: Format-specific triage."""
    from pipeline.phase3_triage import run_phase3

    cat_list = [c.strip() for c in categories.split(",")] if categories else None

    def wrapper(db, config, plog, **kw):
        return run_phase3(
            db,
            config,
            plog,
            workers=workers,
            categories=cat_list,
            db_path=str(get_db_path(click.get_current_context())),
        )

    _run_phase(click.get_current_context(), "triage", wrapper)


@main.command()
def dedup() -> None:
    """Phase 4: Exact deduplication."""
    from pipeline.phase4_dedup import run_phase4

    def wrapper(db, config, plog, **kw):
        return run_phase4(db, config, plog)

    _run_phase(click.get_current_context(), "dedup", wrapper)


@main.command()
@click.option("--model", default=None, help="Ollama model tag")
@click.option("--reinfer", is_flag=True, help="Re-infer even if prompt hash matches")
@click.option("--max-folders", type=int, default=None, help="Limit folders (for testing)")
def folder_meta(model: str | None, reinfer: bool, max_folders: int | None) -> None:
    """Phase 5: Folder semantic inference via Ollama."""
    from pipeline.phase5_folder_meta import run_phase5

    def wrapper(db, config, plog, **kw):
        return run_phase5(db, config, plog, model=model, reinfer=reinfer, max_folders=max_folders)

    _run_phase(click.get_current_context(), "folder_meta", wrapper)


@main.command()
@click.option("--output", type=click.Path(), default=None, help="Output path for report")
def report(output: str | None) -> None:
    """Phase 6: Generate aggregates and markdown report."""
    from pipeline.phase6_report import run_phase6

    def wrapper(db, config, plog, **kw):
        return run_phase6(db, config, plog, output_path=output)

    _run_phase(click.get_current_context(), "report", wrapper)


@main.command()
@click.option("--target", type=click.Choice(["file", "folder"]), required=True)
@click.option("--id", "item_id", type=int, required=True)
@click.option("--reason", required=True)
def exclude(target: str, item_id: int, reason: str) -> None:
    """Phase 7: Mark a file or folder as excluded."""
    from pipeline.phase7_exclude import run_phase7_exclude

    config = get_config(click.get_current_context())
    db = open_db(get_db_path(click.get_current_context()))
    run_phase7_exclude(db, config, target, item_id, reason)
    click.echo(f"Marked {target} {item_id} as excluded: {reason}")


@main.command()
@click.option("--from-file", "csv_path", type=click.Path(exists=True), default=None)
@click.option("--target", type=click.Choice(["file", "folder"]), default=None)
@click.option("--id", "item_id", type=int, default=None)
@click.option("--reason", default=None)
def exclude_batch(
    csv_path: str | None, target: str | None, item_id: int | None, reason: str | None
) -> None:
    """Phase 7: Batch exclude from CSV."""
    from pipeline.phase7_exclude import run_phase7_exclude_batch

    config = get_config(click.get_current_context())
    db = open_db(get_db_path(click.get_current_context()))
    if csv_path:
        applied, skipped = run_phase7_exclude_batch(db, config, Path(csv_path))
        msg = f"Processed exclusions from {csv_path}"
        if applied:
            msg += f" ({applied} applied"
        if skipped:
            msg += f", {skipped} skipped (not found)"
        if applied or skipped:
            msg += ")"
        click.echo(msg)
    elif target and item_id and reason:
        from pipeline.phase7_exclude import run_phase7_exclude

        run_phase7_exclude(db, config, target, item_id, reason)
        click.echo(f"Marked {target} {item_id} as excluded: {reason}")
    else:
        click.echo("Error: provide --from-file CSV or --target/--id/--reason", err=True)
        sys.exit(1)


@main.command()
@click.option("--by", required=True)
def approve(by: str) -> None:
    """Final sign-off on pre-flight."""
    db = open_db(get_db_path(click.get_current_context()))
    from pipeline.helpers import now_iso

    db["config"].insert({"key": "preflight_approved_by", "value": by}, replace=True)
    db["config"].insert({"key": "preflight_completed_at", "value": now_iso()}, replace=True)
    click.echo(f"Pre-flight approved by: {by}")


@main.command()
def status() -> None:
    """Print per-phase status as JSON."""
    db = open_db(get_db_path(click.get_current_context()))
    config_row = next(iter(db.query("SELECT value FROM config WHERE key='corpus_root'")), None)
    corpus_root = config_row["value"] if config_row else "unknown"

    phases = {}
    for row in db.query(
        "SELECT phase, started_at, status, files_processed, files_failed "
        "FROM pipeline_run WHERE id IN (SELECT MAX(id) FROM pipeline_run GROUP BY phase)"
    ):
        phases[row["phase"]] = {
            "last_run": row["started_at"],
            "status": row["status"],
            "files_processed": row["files_processed"],
            "files_failed": row["files_failed"],
        }

    approved = next(iter(db.query("SELECT value FROM config WHERE key='preflight_approved_by'")), None)

    result = {
        "corpus_root": corpus_root,
        "phases": phases,
        "approved": approved is not None,
    }
    click.echo(json.dumps(result, indent=2))


@main.command("serve-datasette")
@click.option("--port", type=int, default=8001)
def serve_datasette(port: int) -> None:
    """Launch Datasette for interactive DB exploration."""
    db_path = get_db_path(click.get_current_context())
    if not db_path.exists():
        click.echo("Error: corpus.db not found. Run phases first.", err=True)
        sys.exit(1)
    click.echo(f"Launching Datasette on port {port}...")
    try:
        subprocess.run(
            [
                "datasette",
                "serve",
                str(db_path),
                "--port",
                str(port),
                "--setting",
                "truncate_cells_html",
                "200",
            ],
            check=True,
        )
    except FileNotFoundError:
        click.echo("Error: datasette not found. Install with: pip install datasette", err=True)
        sys.exit(1)


@main.command()
def run_all() -> None:
    """Run all phases in sequence (convenience command)."""
    ctx = click.get_current_context()
    db_path = get_db_path(ctx)
    if not db_path.exists():
        click.echo("Error: corpus.db not found. Run 'init' first.", err=True)
        sys.exit(1)

    # Clean up any stale model runs from previous crashed processes
    db = open_db(db_path)
    cleaned = cleanup_stale_runs(db)
    if cleaned:
        click.echo(f"Cleaned up {cleaned} stale model run(s)")

    for cmd_name in ["archives", "walk", "identify", "triage", "dedup", "folder-meta", "report"]:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Running: {cmd_name}")
        click.echo(f"{'=' * 60}")
        try:
            ctx.forward(main.commands[cmd_name])
        except SystemExit as e:
            if e.code != 0:
                click.echo(f"Phase {cmd_name} failed. Stopping.", err=True)
                raise


def get_app_config(ctx: click.Context):
    """Load AppConfig from config.toml in project root."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        click.echo(f"Error: config.toml not found at {config_path}. Run 'init' first.", err=True)
        ctx.exit(2)
    try:
        return load_app_config(config_path)
    except Exception as e:
        click.echo(f"Error loading config.toml: {e}", err=True)
        ctx.exit(2)


def _run_app_phase(ctx: click.Context, phase_name: str, phase_fn, **kwargs):
    """Common wrapper for post-preflight phase execution."""
    cfg = get_app_config(ctx)
    db_path = get_db_path(ctx)
    if not is_db_initialized(db_path):
        click.echo("Error: database not initialized. Run 'init' first.", err=True)
        ctx.exit(1)
    db = open_db(db_path)

    from pipeline.logging import PhaseLogger

    plog = PhaseLogger(phase_name, cfg.cache_root_path)

    try:
        result = phase_fn(db, cfg, plog, **kwargs)
        return result
    except Exception as e:
        plog.error(f"Phase {phase_name} failed: {e}")
        raise


@main.command()
@click.option("--workers", type=int, default=None, help="Parallel extraction workers")
@click.option("--strategy", default=None, help="Filter to a single extract_strategy")
@click.option("--limit", type=int, default=None, help="Process at most N files")
@click.option("--reextract", is_flag=True, help="Re-process files already extracted")
def extract(workers: int | None, strategy: str | None, limit: int | None, reextract: bool) -> None:
    """Phase 8: Text extraction from documents."""
    from rag.phase8_extract import run_extract

    def wrapper(db, cfg, plog, **kw):
        return run_extract(
            db, cfg, plog=plog, workers=workers, strategy=strategy, limit=limit, reextract=reextract
        )

    result = _run_app_phase(click.get_current_context(), "extract", wrapper)
    click.echo(
        f"Extracted: {result.get('files_processed', 0)}, Failed: {result.get('files_failed', 0)}"
    )


@main.command()
@click.option("--workers", type=int, default=1, help="Parallel transcription workers")
@click.option("--folder", default=None, help="Filter by folder rel_path prefix")
@click.option("--limit", type=int, default=None, help="Transcribe at most N files")
@click.option("--model", default=None, help="Override transcription model")
def transcribe(workers: int, folder: str | None, limit: int | None, model: str | None) -> None:
    """Phase 8b: Audio/video transcription."""
    from rag.phase8b_transcribe import run_transcribe

    def wrapper(db, cfg, plog, **kw):
        return run_transcribe(
            db, cfg, plog=plog, workers=workers, folder=folder, limit=limit, model=model
        )

    result = _run_app_phase(click.get_current_context(), "transcribe", wrapper)
    click.echo(
        f"Transcribed: {result.get('files_processed', 0)}, Failed: {result.get('files_failed', 0)}"
    )


@main.command()
@click.option("--model", default=None, help="Override summarization model")
@click.option("--limit", type=int, default=None, help="Summarize at most N files")
@click.option("--resummarize", is_flag=True, help="Re-summarize even if already done")
def summarize(model: str | None, limit: int | None, resummarize: bool) -> None:
    """Phase 9: Document summarization."""
    from rag.phase9_summarize import run_summarize

    def wrapper(db, cfg, plog, **kw):
        return run_summarize(db, cfg, limit=limit, resummarize=resummarize)

    result = _run_app_phase(click.get_current_context(), "summarize", wrapper)
    click.echo(f"Summarized: {result[0]}, Failed: {result[1]}")


@main.command()
@click.option("--chunker", default=None, help="Chunker algorithm name")
@click.option("--window", type=int, default=None, help="Sentence window size")
@click.option("--rechunk", is_flag=True, help="Delete and re-chunk existing chunks")
def chunk(chunker: str | None, window: int | None, rechunk: bool) -> None:
    """Phase 10: Sentence-window chunking."""
    from rag.phase10_chunk import run_chunk

    def wrapper(db, cfg, plog, **kw):
        return run_chunk(db, cfg, chunker=chunker, window=window, rechunk=rechunk)

    result = _run_app_phase(click.get_current_context(), "chunk", wrapper)
    click.echo(f"Chunks created: {result[0]}, Files processed: {result[1]}")


@main.command()
@click.option("--model", default=None, help="Override contextual retrieval model")
@click.option("--limit", type=int, default=None, help="Process at most N chunks")
@click.option("--regenerate", is_flag=True, help="Re-generate even if already done")
def context(model: str | None, limit: int | None, regenerate: bool) -> None:
    """Phase 10.5: Contextual retrieval augmentation."""
    from rag.phase10_5_context import run_context, validate_and_clean

    def wrapper(db, cfg, plog, **kw):
        return run_context(db, cfg, limit=limit, regenerate=regenerate)

    result = _run_app_phase(click.get_current_context(), "context", wrapper)
    click.echo(f"Contexts generated: {result[0]}, Failed: {result[1]}")

    # Post-processing quality validation
    click.echo("Running context quality validation...")
    db_path = get_db_path(click.get_current_context())
    from pipeline.db import open_db
    db = open_db(db_path)
    validation = validate_and_clean(db)
    if validation["cleared_for_rerun"] > 0:
        click.echo(
            f"  Issues found: thinking_leaked={validation['thinking_leaked']}, "
            f"thinking_malformed={validation['thinking_malformed']}, "
            f"too_short={validation['too_short']}"
        )
        click.echo(
            f"  Cleared for reprocessing: {validation['cleared_for_rerun']} "
            f"(remaining with context: {validation['remaining_with_context']})"
        )
        click.echo("  Re-run 'rag context' to regenerate cleared entries.")
    else:
        click.echo("  All contexts passed validation.")


@main.command()
@click.option("--model", default=None, help="Override embedding model")
@click.option("--collections", default=None, help="Comma-separated: chunks,summaries,folders")
@click.option("--reembed", is_flag=True, help="Re-embed even if already done")
def embed(model: str | None, collections: str | None, reembed: bool) -> None:
    """Phase 11: Embedding generation."""
    from rag.phase11_embed import run_embed

    def wrapper(db, cfg, plog, **kw):
        collist = [c.strip() for c in collections.split(",")] if collections else None
        return run_embed(db, cfg, model=model, collections=collist, reembed=reembed)

    result = _run_app_phase(click.get_current_context(), "embed", wrapper)
    click.echo(f"Embeddings created: {result}")


@main.command("embed-switch-to")
@click.option("--model", required=True, help="New embedding model name")
@click.option("--suffix", default=None, help="New collection suffix (default: derived from model)")
def embed_switch_to(model: str, suffix: str | None) -> None:
    """Switch to a new embedding model atomically."""
    from rag.phase11_embed import run_embed_switch_to

    cfg = get_app_config(click.get_current_context())
    db = open_db(get_db_path(click.get_current_context()))
    run_embed_switch_to(db, cfg, PROJECT_ROOT / "config.toml", model_name=model, suffix=suffix)


@main.command("embed-gc")
def embed_gc() -> None:
    """Garbage collect superseded embedding collections."""
    from rag.phase11_embed import run_embed_gc

    cfg = get_app_config(click.get_current_context())
    db = open_db(get_db_path(click.get_current_context()))
    run_embed_gc(db, cfg)


@main.command("embed-list")
def embed_list() -> None:
    """List all embedding collections and their status."""
    from rag.phase11_embed import run_embed_list

    cfg = get_app_config(click.get_current_context())
    db = open_db(get_db_path(click.get_current_context()))
    run_embed_list(db, cfg)


@main.command()
@click.argument("question")
@click.option("--folder", default=None, help="Folder filter path prefix")
@click.option("--top-k", type=int, default=None, help="Number of chunks to return")
@click.option("--no-rerank", is_flag=True, help="Disable reranking")
@click.option("--no-augment", is_flag=True, help="Disable contextual augmentation")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def query(
    question: str,
    folder: str | None,
    top_k: int | None,
    no_rerank: bool,
    no_augment: bool,
    as_json: bool,
) -> None:
    """Phase 12: Query the knowledge base."""
    import chromadb

    from rag.generation import generate_answer
    from rag.retrieval import retrieve

    cfg = get_app_config(click.get_current_context())
    db = open_db(get_db_path(click.get_current_context()))
    chroma_client = chromadb.PersistentClient(path=str(cfg.chroma_root_path))

    folder_filter = {"path_prefix": folder} if folder else None

    hits, retrieval_metrics = retrieve(
        question,
        db,
        chroma_client,
        str(cfg.chroma_root_path),
        cfg,
        folder_filter=folder_filter,
        top_k_chunks=top_k,
        use_reranker=not no_rerank,
        use_augmentation=not no_augment,
    )

    result = generate_answer(
        question, hits, cfg, prompt_template=str(PROJECT_ROOT / "prompts" / "generation_v1.txt")
    )

    if as_json:
        output = {
            "answer": result["answer"],
            "citations": result["citations"],
            "metrics": {**retrieval_metrics, "generation_ms": result.get("generation_ms", 0)},
            "models": {
                "embedding": cfg.models.embedding.name,
                "generation": cfg.models.generation.name,
            },
        }
        click.echo(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        click.echo(result["answer"])
        if result.get("citations"):
            click.echo("\n--- Citations ---")
            for c in result["citations"]:
                click.echo(
                    f"  {c['marker']} {c['rel_path']}"
                    + (f" p.{c['page']}" if c.get("page") else "")
                )


@main.command()
@click.option("--host", default=None, help="Override API host")
@click.option("--port", type=int, default=None, help="Override API port")
@click.option("--reload", is_flag=True, help="Enable auto-reload (development)")
def serve(host: str | None, port: int | None, reload: bool) -> None:  # noqa: F811
    """Phase 13: Start the FastAPI server."""
    import uvicorn

    cfg = get_app_config(click.get_current_context())
    h = host or cfg.api.host
    p = port or cfg.api.port
    click.echo(f"Starting API server on {h}:{p}...")
    uvicorn.run("api.main:app", host=h, port=p, reload=reload)


@main.command()
@click.option("--host", default=None, help="Override UI host")
@click.option("--port", type=int, default=None, help="Override UI port")
def ui(host: str | None, port: int | None) -> None:
    """Phase 14: Start the Gradio chat UI."""
    cfg = get_app_config(click.get_current_context())
    from ui.gradio_app import launch_ui

    h = host or cfg.ui.host
    p = port or cfg.ui.port
    click.echo(f"Starting Gradio UI on {h}:{p}...")
    launch_ui(cfg, server_name=h, server_port=p)


@main.group()
def eval() -> None:
    """Evaluation harness."""
    pass


@eval.command()
@click.option("--question", required=True)
@click.option("--expects", required=True, help="Comma-separated file IDs")
@click.option(
    "--category",
    default=None,
    type=click.Choice(["lookup", "synthesis", "scoped", "negative", "bilingual"]),
)
@click.option("--lang", default=None)
def add(question: str, expects: str, category: str | None, lang: str | None) -> None:
    """Add a question to the eval set."""
    from eval.runner import add_question as _add

    db = open_db(get_db_path(click.get_current_context()))
    file_ids = [int(x.strip()) for x in expects.split(",")]
    qid = _add(db, question, file_ids, category=category, lang=lang)
    click.echo(f"Added question #{qid}")


@eval.command("run")
@click.option("--embedding", default=None, help="Override embedding model")
@click.option("--generation", default=None, help="Override generation model")
@click.option("--reranker", default=None, help="Override reranker model")
@click.option("--no-augment", is_flag=True, help="Disable contextual augmentation")
def eval_run(
    embedding: str | None, generation: str | None, reranker: str | None, no_augment: bool
) -> None:
    """Run evaluation against all questions."""
    from eval.runner import run_eval

    cfg = get_app_config(click.get_current_context())
    db = open_db(get_db_path(click.get_current_context()))
    summary = run_eval(
        db,
        cfg,
        embedding_model=embedding,
        generation_model=generation,
        reranker_model=reranker,
        no_augment=no_augment,
    )
    click.echo(json.dumps(summary, indent=2))


@eval.command()
@click.option("--last", type=int, default=None, help="Show last N eval runs")
@click.option("--compare", is_flag=True, help="Show side-by-side comparison")
def report(last: int | None, compare: bool) -> None:  # noqa: F811
    """Show evaluation report."""
    from eval.runner import eval_report as _report

    db = open_db(get_db_path(click.get_current_context()))
    output = _report(db, last_n=last, compare=compare)
    click.echo(output)


@main.group()
def models() -> None:
    """Model management commands."""
    pass


@models.command()
def list() -> None:
    """List active models and collection state."""
    from rag.phase11_embed import run_embed_list

    cfg = get_app_config(click.get_current_context())
    db = open_db(get_db_path(click.get_current_context()))
    run_embed_list(db, cfg)


@models.command()
@click.option("--all", "pull_all", is_flag=True, help="Also pull all alternatives")
def pull(pull_all: bool) -> None:
    """Pull every model named in config."""
    cfg = get_app_config(click.get_current_context())
    model_names = {cfg.models.embedding.name}
    if pull_all:
        for alt in cfg.models.alternatives.values():
            model_names.add(alt.name)
    model_names.add(cfg.models.summarization.name)
    model_names.add(cfg.models.generation.name)
    if cfg.models.contextual_retrieval.enabled:
        model_names.add(cfg.models.contextual_retrieval.name)

    import subprocess

    for name in sorted(model_names):
        click.echo(f"Pulling {name}...")
        subprocess.run(["ollama", "pull", name], check=True)


@models.command()
def check() -> None:
    """Verify every configured model is present in Ollama."""
    import subprocess

    cfg = get_app_config(click.get_current_context())
    model_names = {
        cfg.models.embedding.name,
        cfg.models.summarization.name,
        cfg.models.generation.name,
    }
    if cfg.models.contextual_retrieval.enabled:
        model_names.add(cfg.models.contextual_retrieval.name)

    all_ok = True
    for name in sorted(model_names):
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if name.split(":")[0] in result.stdout:
            click.echo(f"  [green]{name}[/green]")
        else:
            click.echo(f"  [red]{name} - NOT INSTALLED[/red]")
            all_ok = False
    if all_ok:
        click.echo("All configured models are present in Ollama.")


if __name__ == "__main__":
    main()
