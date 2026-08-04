"""CLI BikeFit (squelette Phase 1)."""

import typer

app = typer.Typer(help="Analyse vidéo de position cycliste (CLM)")


@app.command()
def analyze(
    video: str,
    view: str = typer.Option("side", help="side | front"),
    bike: str = typer.Option("mon_clm", help="Nom de la fiche vélo dans config/bikes/"),
) -> None:
    """Analyse une vidéo et produit métriques + recommandations."""
    typer.echo(f"[Phase 1 à implémenter] video={video} view={view} bike={bike}")


if __name__ == "__main__":
    app()
