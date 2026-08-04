# BikeFit CLM

Analyse vidéo de position cycliste (contre-la-montre) par estimation de pose.
Filme-toi sur home trainer (profil + face), obtiens des recommandations de
réglage chiffrées en mm : hauteur/recul de selle, prolongateurs, poste de pilotage.

## Démarrage rapide

```bash
# Prérequis : Python 3.11+ et uv (https://docs.astral.sh/uv/)
uv sync --all-extras

# Analyser une vidéo de profil
python -m bikefit ma_video_profil.mp4 --side right

# Avec profil cible explicite et rapport HTML nommé
python -m bikefit ma_video.mp4 --side right --profile triathlon_XL --report rapport.html
```

## Pipeline

```
vidéo → [ingest] → [pose MediaPipe + cache Parquet] → [kinematics] → [recommend] → [rapport HTML]
```

1. **Ingest** — lit la vidéo (OpenCV), produit des frames horodatées
2. **Pose** — MediaPipe `pose_landmarker_heavy` (Tasks API), résultat mis en cache Parquet
3. **Kinematics** — filtre Savitzky-Golay, détection cycles (extrema cheville), 5 angles vectorisés, agrégation sur N cycles
4. **Recommend** — moteur de règles : angles hors cible → delta mm par réglage
5. **Report** — rapport HTML interactif (Plotly) avec métriques, graphes et recommandations

## Métriques calculées

| Angle | Point du cycle | Interprétation |
|---|---|---|
| Genou extension | PMB | Hauteur de selle |
| Genou flexion | PMH | Recul de selle |
| Hanche | PMH | Reach / hauteur prolongateurs |
| Tronc / horizontale | Moyenne cycle | Aérodynamique |
| Coude | Moyenne cycle | Longueur prolongateurs |
| Cheville | PMB | Technique de pédalage / cales |

## Résultats obtenus (vidéo de référence, 1506 frames, 30 fps)

- Détection : **100%** — 59 cycles
- Genou extension PMB : **150.1° ± 2.2°** (cible 140-150°)
- Tronc : **18.0° ± 0.5°**
- Coude : **115.7° ± 0.9°**

## Options CLI

```
python -m bikefit VIDEO [OPTIONS]

Arguments:
  VIDEO              Vidéo de profil (mp4/mov)

Options:
  --side             right | left (côté filmé, défaut: right)
  --bike             Fiche vélo dans config/bikes/ (défaut: giant_trinity_2019)
  --profile          Profil cible : triathlon_XL | clm_long | clm_court (défaut: triathlon_XL)
  --report PATH      Rapport HTML de sortie (défaut: même dossier que la vidéo)
  --output PATH      Métriques JSON de sortie
  --no-cache         Force le recalcul des landmarks MediaPipe
```

## Structure du projet

```
bikefit/
├── config/
│   ├── targets_tt.yaml          # angles cibles par profil (clm_court/long, triathlon_XL)
│   ├── bikes/                   # fiches vélo personnelles (gitignorées)
│   └── rider/                   # mensurations personnelles (gitignorées)
├── models/                      # modèles MediaPipe .task (gitignorés)
├── cache/landmarks/             # cache Parquet par vidéo (gitignorés)
├── scripts/
│   └── spike_bikefit.py         # script autonome Phase 0
└── src/bikefit/
    ├── ingest/                  # VideoReader
    ├── pose/                    # PoseEstimator (Protocol) + MediaPipe + LandmarkCache
    ├── kinematics/              # filter, cycles, angles, aggregate, metrics
    ├── equipment/               # BikeProfile, BikeSetup, loader YAML
    ├── recommend/               # moteur de règles → FitReport
    ├── report/                  # rapport HTML Jinja2 + Plotly
    └── cli.py                   # Typer CLI
```

## Tests

```bash
python -m pytest        # 20 tests (angles sur squelettes synthétiques + moteur de recommandations)
```

## Documentation

- [Plan de développement & architecture](docs/plan-developpement.md)
- [CLAUDE.md](CLAUDE.md) — contexte pour Claude Code

## Statut

- [x] Phase 0 — spike MediaPipe (Tasks API v1.0)
- [x] Phase 1 — pipeline ingest → pose → kinematics → CLI, jalon ±2° atteint
- [x] Phase 2 — moteur de règles, fiche vélo, rapport HTML
- [ ] Phase 3 — vue de face, BikeSetup versionné, historique SQLite
- [ ] Phase 4 — UI Streamlit (optionnel)
