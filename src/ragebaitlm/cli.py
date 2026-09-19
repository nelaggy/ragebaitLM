"""Command line interface for ragebaitLM."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import analyze as analyze_mod
from .adapters import ADAPTERS, get_adapter
from .pipeline import build_engine
from .pipeline import sync as run_sync
from .report import build_report
from .store import Store

app = typer.Typer(
    add_completion=False,
    help="Analyse coding-agent sessions for user mood (joy ↔ rage).",
)
console = Console()

DEFAULT_DB = Path("data/ragebaitlm.db")
ENGINES = ("hybrid", "transformer", "lexicon")


def _parse_since(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter("--since must be ISO date, e.g. 2026-01-01") from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@app.command("list-harnesses")
def list_harnesses(home: Optional[Path] = typer.Option(None, help="Override home dir")):
    """Show each harness's default path and whether it exists."""
    table = Table(title="Harnesses")
    table.add_column("harness")
    table.add_column("default path")
    table.add_column("exists")
    for name in ADAPTERS:
        adapter = get_adapter(name, home)
        exists = adapter.default_path.exists()
        table.add_row(name, str(adapter.default_path), "yes" if exists else "no")
    console.print(table)


@app.command()
def sync(
    harness: list[str] = typer.Option(
        ["all"], "--harness", "-H", help=f"Harnesses: {', '.join(ADAPTERS)} or 'all'"
    ),
    path: Optional[list[Path]] = typer.Option(
        None, "--path", "-p", help="Override harness path(s). Use once per harness."
    ),
    since: Optional[str] = typer.Option(None, "--since", help="Only sessions after this date"),
    db: Path = typer.Option(DEFAULT_DB, "--db", help="Output SQLite database"),
    engine: str = typer.Option("hybrid", "--engine", help=f"One of {', '.join(ENGINES)}"),
    cache_text: bool = typer.Option(
        False, "--cache-text", help="Also store raw message text (local experimentation)"
    ),
    home: Optional[Path] = typer.Option(None, help="Override home dir for harness discovery"),
):
    """Import session logs, score user messages, and store stats."""
    if engine not in ENGINES:
        raise typer.BadParameter(f"--engine must be one of {', '.join(ENGINES)}")
    names = list(ADAPTERS) if "all" in harness else harness
    unknown = [n for n in names if n not in ADAPTERS]
    if unknown:
        raise typer.BadParameter(f"Unknown harness(es): {unknown}")

    try:
        config = build_engine(engine)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    if config.note:
        console.print(f"[yellow]{config.note}[/yellow]")

    since_ms = _parse_since(since)
    store = Store(db, cache_text=cache_text)
    store.init_schema()

    total = {"sessions": 0, "messages": 0, "human": 0}
    try:
        for name in names:
            adapter = get_adapter(name, home)
            override = list(path) if path else None
            console.print(f"[bold]→ {name}[/bold] ({adapter.default_path})")
            result = run_sync(store, adapter, config, paths=override, since=since_ms)
            for key in total:
                total[key] += result.get(key, 0)
            console.print(
                f"  {result['sessions']} sessions · {result['human']} user messages · "
                f"{result['messages']} messages"
            )
    finally:
        store.close()

    console.print(
        f"[green]done[/green] {total['sessions']} sessions, "
        f"{total['human']} user messages → {db}"
    )


@app.command()
def analyze(
    db: Path = typer.Option(DEFAULT_DB, "--db", help="SQLite database"),
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON"),
    top: int = typer.Option(10, "--top", help="Rows per table"),
):
    """Print summary statistics and model mood."""
    store = Store(db)
    try:
        result = analyze_mod.analyze(store)
    finally:
        store.close()

    if as_json:
        console.print_json(analyze_mod.dumps(result))
        return

    overall = result["overall"]
    if not overall.get("n_messages"):
        console.print("[yellow]No scored messages. Run `ragebaitlm sync` first.[/yellow]")
        return

    mean_mood = overall["mean_mood"]
    color = "red" if mean_mood < 0 else "green"
    console.print(
        f"[bold]{overall['n_messages']}[/bold] user messages across "
        f"[bold]{overall['n_sessions']}[/bold] sessions · "
        f"mean mood [{color}]{mean_mood * 100:+.1f}[/{color}] · "
        f"[red]rage {overall['pct_rage'] * 100:.1f}%[/red] · "
        f"[green]joy {overall['pct_joy'] * 100:.1f}%[/green]"
    )
    if overall.get("n_subagent_sessions"):
        console.print(
            f"[dim]{overall['n_subagent_sessions']} subagent sessions excluded "
            f"({overall['n_stored_sessions']} total stored)[/dim]"
        )

    for title, rows in (
        ("By preceding model", result["by_model"]),
        ("By harness", result["by_harness"]),
    ):
        table = Table(title=title)
        table.add_column("key")
        table.add_column("messages", justify="right")
        table.add_column("mean mood", justify="right")
        table.add_column("median mood", justify="right")
        table.add_column("% rage", justify="right")
        table.add_column("% joy", justify="right")
        for row in rows[:top]:
            table.add_row(
                row["key"],
                str(row["n"]),
                f"{row['mean'] * 100:+.1f}",
                f"{row['median'] * 100:+.1f}",
                f"{row['pct_rage'] * 100:.0f}%",
                f"{row['pct_joy'] * 100:.0f}%",
            )
        console.print(table)


@app.command()
def report(
    db: Path = typer.Option(DEFAULT_DB, "--db", help="SQLite database"),
    out: Path = typer.Option(Path("data/report.html"), "--out", "-o", help="Output HTML"),
):
    """Build a self-contained HTML report."""
    store = Store(db)
    try:
        result = analyze_mod.analyze(store)
    finally:
        store.close()
    path = build_report(result, out)
    console.print(f"[green]report[/green] → {path}")


if __name__ == "__main__":
    app()
