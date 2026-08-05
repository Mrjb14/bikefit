"""Point d'entrée — injecte 'analyze' si aucun sous-commande n'est fournie.

Permet d'appeler :
    python -m bikefit video.mp4 --side right        (rétro-compatible)
    python -m bikefit analyze   video.mp4 --side right  (explicite)
    python -m bikefit history
    python -m bikefit compare 1 2
"""

import sys

_SUBCOMMANDS = {"analyze", "history", "compare", "measure-body", "measure-setup"}


def _inject_default_command() -> None:
    args = sys.argv[1:]
    if not args:
        sys.argv.append("--help")
        return
    first = args[0]
    # Si le premier argument n'est pas un sous-commande connu ni une option,
    # c'est sûrement un chemin vidéo → on préfixe avec "analyze".
    if not first.startswith("-") and first not in _SUBCOMMANDS:
        sys.argv.insert(1, "analyze")


_inject_default_command()

from bikefit.cli import app  # noqa: E402

app()
