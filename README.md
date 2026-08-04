# BikeFit CLM

Analyse vidéo de position cycliste (contre-la-montre) par estimation de pose.
Filme-toi sur home trainer (profil + face), obtiens des recommandations de
réglage chiffrées en mm : hauteur/recul de selle, prolongateurs, poste de pilotage.

## Démarrage rapide

```bash
# Prérequis : Python 3.11+ et uv (https://docs.astral.sh/uv/)
uv sync --all-extras

# Spike Phase 0 : valider la détection de pose sur ta vidéo
uv run python scripts/spike_bikefit.py ma_video_profil.mp4 --side right
```

## Documentation

- [Plan de développement & architecture](docs/plan-developpement.md)
- [CLAUDE.md](CLAUDE.md) — contexte pour Claude Code

## Statut

🚧 Phase 0 (spike technique) — voir le plan pour la feuille de route complète.
