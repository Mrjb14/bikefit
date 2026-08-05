# BikeFit CLM

[![CI](https://github.com/Mrjb14/bikefit/actions/workflows/ci.yml/badge.svg)](https://github.com/Mrjb14/bikefit/actions/workflows/ci.yml)

Analyse vidéo de position cycliste (contre-la-montre) par estimation de pose.
Filme-toi sur home trainer (profil + face), obtiens des recommandations de
réglage chiffrées en mm : hauteur/recul de selle, prolongateurs, poste de pilotage.

## Démarrage rapide

```bash
# Prérequis : Python 3.11+ et uv (https://docs.astral.sh/uv/)
uv sync --all-extras

# Analyser une vidéo de profil
python -m bikefit ma_video.mp4 --side right

# Avec profil cible, calibration ArUco et sauvegarde de session
python -m bikefit ma_video.mp4 --side right --profile triathlon_XL \
    --marker-size 100 --save-session --report rapport.html
```

## Pipeline

```
vidéo ──► [ingest] ──► [pose MediaPipe + cache Parquet] ──► [calibration ArUco]
                                                                      │
                                  [rapport HTML] ◄── [recommend] ◄── [kinematics]
                                       │
                                  [session SQLite]
```

| Étape | Module | Description |
|---|---|---|
| Ingest | `ingest/` | Lit la vidéo (OpenCV), produit des frames horodatées |
| Pose | `pose/` | MediaPipe `pose_landmarker_heavy` (Tasks API), cache Parquet |
| Calibration | `calibration/` | Détecte un marqueur ArUco → `mm_per_pixel` (optionnel) |
| Kinematics | `kinematics/` | Filtre SG, détection cycles, 5 angles vectorisés, agrégation |
| Recommend | `recommend/` | Règles explicites : angles hors cible → delta mm par réglage |
| Report | `report/` | Rapport HTML interactif (Plotly), 5 graphes d'angles, schéma SVG |
| Storage | `storage/` | Sessions SQLite : historique, comparaison avant/après |

## Métriques calculées

| Angle | Point du cycle | Réglage associé |
|---|---|---|
| Genou extension | PMB | Hauteur de selle |
| Genou flexion | PMH | Recul de selle |
| Hanche | PMH | Hauteur prolongateurs |
| Tronc / horizontale | Moyenne cycle | Aérodynamique |
| Coude | Moyenne cycle | Longueur prolongateurs |
| Cheville | PMB | Technique / cales |

## Résultats obtenus (vidéo de référence, 1506 frames, 30 fps)

- Détection : **100%** — 59 cycles
- Genou extension PMB : **150.1° ± 2.2°** (cible 140-150°)
- Tronc : **18.0° ± 0.5°**
- Coude : **115.7° ± 0.9°**

## Options CLI

### `analyze` — analyse une vidéo (commande par défaut)

```
python -m bikefit VIDEO [OPTIONS]
python -m bikefit analyze VIDEO [OPTIONS]

Options:
  --side TEXT          right | left — côté filmé (défaut: right)
  --bike TEXT          Fiche vélo dans config/bikes/ (défaut: giant_trinity_2019)
  --profile TEXT       triathlon_XL | clm_long | clm_court (défaut: triathlon_XL)
  --report PATH        Rapport HTML de sortie
  --output PATH        Métriques JSON de sortie
  --marker-size FLOAT  Taille du marqueur ArUco en mm (ex. 100) — active la calibration
  --save-session       Sauvegarde la session dans ~/.bikefit/sessions.db
  --db PATH            Chemin alternatif pour la base SQLite
  --no-cache           Force le recalcul des landmarks MediaPipe
```

### `history` — liste les sessions

```
python -m bikefit history [--limit N] [--db PATH]
```

### `compare` — compare deux sessions avant/après réglage

```
python -m bikefit compare ID1 ID2 [--db PATH]
```

Affiche un tableau des deltas d'angles et les recommandations restantes.

## Calibration ArUco (optionnelle)

Colle un marqueur ArUco `DICT_4X4_50 ID=0` sur le cadre (tube de selle ou diagonal,
côté caméra). Le fichier prêt à imprimer est dans `docs/aruco_marker_100mm.png`.

```bash
# Imprimer à exactement 100 mm de côté (échelle 100%, ne pas cocher "ajuster à la page")
# Puis analyser avec :
python -m bikefit video.mp4 --side right --marker-size 100
# → Calibration fiable : 0.4312 mm/px (47/60 frames)
```

La calibration est stockée dans `FitMetrics.calibration` et sauvegardée avec la session.
Elle permettra à terme de mesurer les réglages du vélo directement depuis la vidéo
(hauteur de selle, reach, stack) sans mètre ruban.

## Structure du projet

```
bikefit/
├── config/
│   ├── targets_tt.yaml          # angles cibles (clm_court/long, triathlon_XL) + coefficients
│   ├── bikes/                   # fiches vélo personnelles (gitignorées)
│   └── rider/                   # mensurations personnelles (gitignorées)
├── docs/
│   ├── plan-developpement.md
│   └── aruco_marker_100mm.png   # marqueur à imprimer (100 mm, DICT_4X4_50 ID=0)
├── models/                      # modèles MediaPipe .task (gitignorés)
├── cache/landmarks/             # cache Parquet par vidéo (gitignorés)
├── scripts/
│   └── spike_bikefit.py         # script autonome Phase 0
└── src/bikefit/
    ├── ingest/                  # VideoReader, TimedFrame
    ├── pose/                    # PoseEstimator (Protocol) + MediaPipe + LandmarkCache
    ├── kinematics/              # filter, cycles, angles, aggregate, FitMetrics
    ├── calibration/             # ArUco — detect_marker_size_px, calibrate_from_video
    ├── equipment/               # BikeProfile, BikeSetup, loader YAML
    ├── recommend/               # moteur de règles → FitReport + Recommendation
    ├── report/                  # rapport HTML Jinja2 + Plotly (dark theme)
    ├── storage/                 # Session + SessionStore (SQLite)
    └── cli.py                   # Typer — analyze / history / compare
```

## Tests

```bash
python -m pytest        # 49 tests — angles, ArUco, recommandations, storage (données synthétiques)
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

## Documentation

- [Plan de développement & architecture](docs/plan-developpement.md)
- [CLAUDE.md](CLAUDE.md) — contexte pour Claude Code

## Statut

- [x] Phase 0 — spike MediaPipe (Tasks API v1.0)
- [x] Phase 1 — pipeline ingest → pose → kinematics → CLI, jalon ±2° atteint
- [x] Phase 2 — moteur de règles, fiche vélo, rapport HTML (coefficients empiriques)
- [x] Phase 3A — calibration ArUco : `mm_per_pixel` détecté, stocké en session *(marqueur non encore imprimé)*
- [x] Phase 3B — sessions SQLite : `history`, `compare`, `--save-session`
- [ ] Phase 3A+ — mesure automatique du vélo (BB → selle, reach, stack) via 2ᵉ marqueur ArUco
- [ ] Phase 3C — vue de face (ESP32-CAM MJPEG) : alignement genoux, largeur coudes
- [ ] Phase 4 — UI Streamlit (optionnel)
