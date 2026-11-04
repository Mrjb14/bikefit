"""Point d'entrée — injecte 'analyze' si aucun sous-commande n'est fournie.

Permet d'appeler :
    python -m bikefit video.mp4 --side right        (rétro-compatible)
    python -m bikefit analyze   video.mp4 --side right  (explicite)
    python -m bikefit history
    python -m bikefit compare 1 2
"""

import sys

from bikefit.cli import app


def subcommand_names() -> set[str]:
    """Noms réels des commandes Typer (jamais une liste à tenir à jour à la main)."""
    return {
        cmd.name or (cmd.callback.__name__.replace("_", "-") if cmd.callback else "")
        for cmd in app.registered_commands
    }


def _inject_default_command() -> None:
    args = sys.argv[1:]
    if not args:
        sys.argv.append("--help")
        return
    first = args[0]
    # Si le premier argument n'est pas un sous-commande connu ni une option,
    # c'est sûrement un chemin vidéo → on préfixe avec "analyze".
    if not first.startswith("-") and first not in subcommand_names():
        sys.argv.insert(1, "analyze")


if __name__ == "__main__":
    _inject_default_command()
    app()
