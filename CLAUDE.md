# CLAUDE.md — Contexte projet pour Claude Code

## Le projet en une phrase
Logiciel d'analyse vidéo de bike fitting pour contre-la-montre : le cycliste se
filme en pédalant (profil + face), le logiciel estime la pose (MediaPipe),
calcule les angles articulaires moyennés sur plusieurs cycles de pédalage, et
recommande des réglages chiffrés en mm (selle, prolongateurs, poste de pilotage).

## Documents de référence
- `docs/plan-developpement.md` : plan complet, architecture, phases — **à lire en premier**.
- `config/bikes/giant_trinity_2019.yaml` : fiche du vélo cible (Giant Trinity Advanced Pro 2, 2019, M).
- `config/rider/mensurations.yaml` : mensurations personnelles + objectif (Ironman 2027-05-21).
- `scripts/spike_bikefit.py` : spike Phase 0 (script autonome, Tasks API MediaPipe v1.0).

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
- `ingest/` : décodage vidéo (OpenCV), itérateur de frames horodatées (`TimedFrame`).
- `pose/` : protocole `PoseEstimator` + `MediaPipePoseEstimator` (Tasks API, `pose_landmarker_heavy`) + `LandmarkCache` (Parquet).
- `kinematics/` : `filter_landmarks` (Savitzky-Golay ~6 Hz), `detect_cycles` (extrema cheville y),
  `compute_angles` (genou, hanche, tronc, coude, cheville — vectorisé NumPy), `aggregate_cycles` → `FitMetrics`.
- `equipment/` : modèles Pydantic `BikeProfile` / `BikeSetup` / `PlagesReglage`, `load_bike_profile(name)`.
- `recommend/` : `run(metrics, bike, profile)` → `FitReport` + `Recommendation` (règles explicites, pas de ML).
  Cibles dans `config/targets_tt.yaml` (profils : `clm_court`, `clm_long`, `triathlon_XL`).
- `report/` : `render_html(metrics, report, angle_df, video_path, output_path)` — Jinja2 + Plotly CDN.
- `storage/` : (Phase 3) sessions SQLite, historique BikeSetup.
- `cli.py` : Typer — `python -m bikefit video.mp4 --side right --profile triathlon_XL`.

## État d'avancement
- [x] Phase 0 : spike MediaPipe Tasks API (`scripts/spike_bikefit.py`). Modèle `pose_landmarker_heavy.task` dans `models/`.
- [x] Phase 1 : pipeline complet ingest → pose → cache → kinematics → CLI.
      Jalon ±2° atteint : 59 cycles, 100% détection, std ≤ 2.2° sur tous les angles sauf cheville.
- [x] Phase 2 : `equipment/` (BikeProfile YAML), `recommend/` (moteur de règles), `report/` (HTML Plotly).
      Calibration ArUco (mm/pixel) **non encore implémentée** — les recommandations mm sont estimées via coefficients de sensibilité.
- [ ] Phase 3 : vue de face, BikeSetup versionné par session, historique SQLite.
- [ ] Phase 4 (option) : UI Streamlit.

## Commandes
```bash
# Environnement
uv sync --all-extras

# Tests (20 tests : angles + recommandations)
python -m pytest

# Lint
uv run ruff check src tests

# Pipeline complet
python -m bikefit video.mp4 --side right --profile triathlon_XL

# Spike Phase 0
python scripts/spike_bikefit.py video.mp4 --side right
```

## Conventions de code
- Python 3.11+, type hints partout, Pydantic v2 pour toute structure de données publique.
- Tests pytest : la géométrie (angles) et les recommandations sont testées sur des données
  synthétiques aux valeurs connues (`tests/`), jamais uniquement sur des vidéos réelles.
- Clé de cache : `md5(str(path.resolve()) + str(mtime) + str(size))[:12]` — rapide, pas de hash contenu.
- Angles vectorisés sur l'ensemble des frames (NumPy), pas de boucle frame par frame.
- Convention y-bas (OpenCV) : PMB = maximum local de `ankle_y`, détecté par `scipy.find_peaks`.
- Pas de dépendance GPU obligatoire : tout tourne sur CPU.

## Matériel de l'utilisateur (contexte capture)
- Tacx Neo 2 (direct drive), smartphone 1080p/30fps, vue profil = source principale.
- ESP32-CAM (basse résolution ~15-25 fps) — éventuelle vue de face (Phase 3), flux MJPEG HTTP.
- Le module ingest doit accepter : fichier vidéo ET flux réseau MJPEG.
- Vidéos non versionnées (trop volumineuses) — gitignorées globalement.
