# CLAUDE.md — Contexte projet pour Claude Code

## Le projet en une phrase
Logiciel d'analyse vidéo de bike fitting pour contre-la-montre : le cycliste se
filme en pédalant (profil + face), le logiciel estime la pose (MediaPipe),
calcule les angles articulaires moyennés sur plusieurs cycles de pédalage, et
recommande des réglages chiffrés en mm (selle, prolongateurs, poste de pilotage).

## Documents de référence
- `docs/plan-developpement.md` : plan complet, architecture, phases — **à lire en premier**.
- `config/bikes/mon_clm.yaml` : fiche du vélo cible (Giant Trinity Advanced Pro 2, 2018, M).
- `scripts/spike_bikefit.py` : spike Phase 0 (script autonome, sera remplacé par le pipeline).

## Principes non négociables
1. **Précision avant tout** : chaque mesure est moyennée sur N cycles de pédalage,
   jamais sur une frame unique. Toute métrique porte un écart-type / indice de confiance.
2. **Cache des landmarks** : la pose estimation d'une vidéo n'est calculée qu'une
   fois, résultat en Parquet dans le cache. Tout le reste du pipeline doit pouvoir
   se rejouer depuis ce cache en < 1 s.
3. **Modules découplés** : ingest → pose → kinematics → recommend → report.
   L'interface `PoseEstimator` est une abstraction (pattern Strategy) : ne jamais
   coupler le code métier à MediaPipe directement.
4. **Unités** : mm et degrés partout. Axe y image vers le bas (convention OpenCV) —
   attention aux signes dans les calculs d'angles.

## Architecture (src/bikefit/)
- `ingest/` : décodage vidéo (OpenCV), normalisation, itérateur de frames horodatées.
- `pose/` : interface `PoseEstimator` + implémentation MediaPipe (`pose_landmarker_heavy`).
- `kinematics/` : filtrage (Savitzky-Golay ~6 Hz), détection cycles (extrema cheville),
  angles (genou, hanche, tronc, coude, cheville), agrégation multi-cycles,
  calibration mm/pixel via ArUco.
- `equipment/` : modèles Pydantic `BikeProfile` / `BikeSetup`, plages de réglage,
  chargement YAML depuis `config/bikes/`.
- `recommend/` : moteur de règles (pas de ML). Consomme FitMetrics + BikeProfile.
  Détecte les butées de réglage. Cibles dans `config/targets_tt.yaml`.
- `report/` : rapport HTML (Jinja2 + Plotly), frames annotées aux points morts.
- `storage/` : cache Parquet des landmarks, sessions SQLite.
- `cli.py` : CLI Typer (`bikefit analyze video.mp4 --view side --bike mon_clm`).

## État d'avancement
- [x] Phase 0 : spike MediaPipe (`scripts/spike_bikefit.py`) — en cours de validation
      sur vidéos réelles.
- [ ] Phase 1 : pipeline profil (ingest, pose, cache, cycles, angles). Jalon : ±2°
      de reproductibilité entre deux captures.
- [ ] Phase 2 : calibration ArUco, moteur de règles, fiche vélo, rapport HTML.
- [ ] Phase 3 : vue de face, BikeSetup versionné, historique SQLite.
- [ ] Phase 4 (option) : UI Streamlit.

## Commandes
```bash
uv sync --all-extras        # installer l'environnement
uv run pytest               # tests
uv run ruff check src tests # lint
uv run bikefit --help       # CLI (à partir de la Phase 1)
uv run python scripts/spike_bikefit.py video.mp4 --side right  # spike
```

## Conventions de code
- Python 3.11+, type hints partout, Pydantic pour toute structure de données publique.
- Tests pytest : la géométrie (angles) est testée sur des squelettes synthétiques
  aux angles connus (`tests/`), jamais uniquement sur des vidéos réelles.
- Les vidéos de test sont courtes (< 5 s) et vivent dans `tests/data/` (Git LFS si > 5 Mo).
- Pas de dépendance GPU obligatoire : tout doit tourner sur CPU.

## Matériel de l'utilisateur (contexte capture)
- Home trainer, smartphone (1080p/60fps, vue profil = source principale),
  ESP32-CAM (basse résolution ~15-25 fps, éventuelle vue de face, flux MJPEG HTTP).
- Le module ingest doit donc accepter : fichier vidéo ET flux réseau MJPEG.
