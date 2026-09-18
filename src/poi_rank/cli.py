"""Typer CLI entry point for the poi-rank pipeline.

`make generate` runs `python -m poi_rank.cli generate`. Later phases (prepare,
features, candidates, train, evaluate, scenarios) are wired in as they're built.
"""

from __future__ import annotations

from pathlib import Path

import typer

from poi_rank.datagen.config import DatagenConfig
from poi_rank.datagen.pipeline import run_generate

app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "datagen.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "synthetic"


@app.callback()
def main() -> None:
    """poi-rank pipeline CLI. Subcommands are added as each build phase lands."""


@app.command()
def generate(
    # typer's documented idiom requires the call in the default; B008 is a false
    # positive here (typer reads the Option() at import time, not per-call).
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to datagen.yaml"),  # noqa: B008
    output_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT_DIR, help="Output directory for data/synthetic/"
    ),
) -> None:
    """Run the full synthetic data-generation pipeline and print a summary report."""
    cfg = DatagenConfig.from_yaml(config_path)
    summary = run_generate(cfg, output_dir)

    counts = summary["counts"]
    typer.echo("=== poi-rank datagen: generation summary ===")
    for key, value in counts.items():
        typer.echo(f"  {key}: {value}")

    typer.echo("\n=== SHA256 of generated files ===")
    for name, digest in sorted(summary["sha256"].items()):
        typer.echo(f"  {name}: {digest}")


if __name__ == "__main__":
    app()
