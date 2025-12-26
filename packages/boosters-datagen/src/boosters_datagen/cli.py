"""CLI for test data generation."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="boosters-datagen",
    help="Generate test data for boosters library.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def xgboost() -> None:
    """Generate XGBoost test cases."""
    from boosters_datagen.xgboost import generate_all

    console.print("[bold]Generating XGBoost test cases...[/bold]")
    generate_all()


@app.command()
def lightgbm() -> None:
    """Generate LightGBM test cases."""
    from boosters_datagen.lightgbm import generate_all

    console.print("[bold]Generating LightGBM test cases...[/bold]")
    generate_all()


@app.command(name="all")
def all_cmd() -> None:
    """Generate all test cases."""
    from boosters_datagen.lightgbm import generate_all as lgb_all
    from boosters_datagen.xgboost import generate_all as xgb_all

    console.print("[bold]Generating all test cases...[/bold]")
    xgb_all()
    lgb_all()
    console.print("\n[green]✓ All test cases generated[/green]")


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
