# CLAUDE.md — Contexte projet pour Claude Code

## Le projet en une phrase
Logiciel d'analyse vidéo de bike fitting pour contre-la-montre : le cycliste se
filme en pédalant (profil + face), le logiciel estime la pose (MediaPipe),
calcule les angles articulaires moyennés sur plusieurs cycles de pédalage, et
recommande des réglages chiffrés en mm (selle, prolongateurs, poste de pilotage).

## Documents de référence
- `docs/plan-developpement.md` : plan complet, architecture, phases — **à lire en premier**.
- `docs/procedure-mesures.md` : procédure de prise de mesures (corps + setup vélo), bootstrap automatique
  (`measure-body`/`measure-setup`) puis validation obligatoire au mètre ruban.
- `config/bikes/giant_trinity_2019.yaml` : fiche du vélo cible (Giant Trinity Advanced Pro 2, 2019, M).
- `config/rider/mensurations.yaml` : mensurations personnelles + objectif (Ironman 2027-05-21).
- `scripts/spike_bikefit.py` : spike Phase 0 (script autonome, Tasks API MediaPipe v1.0).
- `docs/aruco_marker_100mm.png` : marqueur ArUco à imprimer (DICT_4X4_50 ID=0, 100 mm).
- `firmware/README.md` : installation du firmware ESP32-CAM (flux MJPEG, vue de face Phase 3C).
- `data/` : vidéos pour les tests manuels du CLI (gitignorées, `*.mp4`).

## Principes non négociables
1. **Précision avant tout** : chaque mesure est moyennée sur N cycles de pédalage,
   jamais sur une frame unique. Toute métrique porte un écart-type / indice de confiance.
2. **Cache des landmarks** : la pose estimation d'une vidéo n'est calculée qu'une
   fois, résultat en Parquet dans le cache. Tout le reste du pipeline doit pouvoir
   se rejouer depuis ce cache en < 1 s.
3. **Modules découplés** : ingest → pose → calibration → kinematics → recommend → report.
   L'interface `PoseEstimator` est une abstraction (pattern Strategy) : ne jamais
   coupler le code métier à MediaPipe directement.
4. **Unités** : mm et degrés partout. Axe y image vers le bas (convention OpenCV) —
   attention aux signes dans les calculs d'angles.

## Architecture (src/bikefit/)
- `ingest/` : `VideoReader` — fichier vidéo (OpenCV) ou flux réseau MJPEG (`http(s)://`/`rtsp://`,
  ex. ESP32-CAM) via parsing manuel des marqueurs JPEG SOI/EOI, itérateur de frames horodatées
  (`TimedFrame`). Flux réseau : timestamps sur horloge murale, pas de fps fiable côté source.
- `pose/` : protocole `PoseEstimator` + `MediaPipePoseEstimator` (Tasks API, `pose_landmarker_heavy`) + `LandmarkCache` (Parquet).
- `kinematics/` : `filter_landmarks` (Savitzky-Golay ~6 Hz), `detect_cycles` (extrema cheville y),
  `compute_angles` (genou, hanche, tronc, coude, cheville — vectorisé NumPy), `aggregate_cycles` → `FitMetrics`.
  `frontal.py` (Phase 3C, vue de face) : `compute_frontal_metrics` (déviation angulaire genou vs droite
  hanche-cheville, largeur coudes px) + `aggregate_frontal_cycles` → champs optionnels de `FitMetrics`.
  `FitMetrics` porte des champs optionnels `calibration: CalibrationResult | None` et plan frontal
  (`genou_deviation_gauche/droit`, `largeur_coudes`).
- `calibration/` : `detect_marker_size_px` (frame unique) + `calibrate_from_video` (vidéo entière,
  médiane sur frames échantillonnées) → `CalibrationResult(mm_per_pixel, n_detected, n_sampled, is_reliable)`.
  Dictionnaire ArUco : `DICT_4X4_50`. Marqueur ID=0.
- `equipment/` : modèles Pydantic `BikeProfile` / `BikeSetup` / `PlagesReglage`, `load_bike_profile(name)`.
- `recommend/` : `run(metrics, bike, profile)` → `FitReport` + `Recommendation` (règles explicites, pas de ML).
  Cibles dans `config/targets_tt.yaml` (profils : `clm_court`, `clm_long`, `triathlon_XL`).
  Coefficients de sensibilité empiriques (°/mm) — seront affinés par calibration ArUco (Phase 3A+).
- `report/` : `render_html(metrics, report, angle_df, video_path, output_path)` — Jinja2 + Plotly CDN.
  Dark theme, 5 graphes d'angles empilés, schéma SVG du vélo de profil + schéma vue de face (Phase 3C,
  roue en tranche, cintre symétrique) avec annotations optionnelles si métriques plan frontal présentes.
- `storage/` : `Session` (Pydantic) + `SessionStore` (SQLite natif `sqlite3`).
  Opérations : `save`, `get`, `list_all`, `latest`, `delete`. Tri par `date DESC, id DESC`.
- `measure/` : bootstrap des YAML personnels (`config/rider/`, `config/bikes/`) depuis photo/vidéo.
  `estimate_body_measurements` (photo debout, MediaPipe IMAGE mode, world_landmarks 3D, mise à l'échelle
  via ratio nez→cheville / taille) ; `estimate_bike_setup` (vidéo + calibration ArUco, fit de cercle de
  Kåsa sur la trajectoire de la cheville → centre BB, puis `hauteur_selle_bb`/`hauteur_guidon_bb`/
  `reach_prolongateurs` par projection sur BB) ; `update_null_fields` (yaml_io) réécrit uniquement les
  champs `null` d'un YAML existant, commentaires préservés. N'écrit jamais un champ déjà renseigné —
  les estimations sont un point de départ (±5-10 % corps, ±20 mm setup), à valider au mètre ruban.
  `render.py` : annotation visuelle OpenCV (squelette + crochets de cotation) sur la photo/frame PMB,
  pour vérifier les estimations avant `--write`.
- `cli.py` : Typer — 5 commandes : `analyze` (défaut), `history`, `compare`, `measure-body`, `measure-setup`.
  `__main__.py` injecte `analyze` si le premier argument n'est pas un sous-commande.

## État d'avancement
- [x] Phase 0 : spike MediaPipe Tasks API (`scripts/spike_bikefit.py`). Modèle `pose_landmarker_heavy.task` dans `models/`.
- [x] Phase 1 : pipeline complet ingest → pose → cache → kinematics → CLI.
      Jalon ±2° atteint : 59 cycles, 100% détection, std ≤ 2.2° sur tous les angles sauf cheville.
- [x] Phase 2 : `equipment/` (BikeProfile YAML), `recommend/` (moteur de règles), `report/` (HTML Plotly).
      Recommandations en mm via coefficients empiriques (ArUco non encore utilisé pour les delta_mm).
- [x] Phase 3A : `calibration/` — détection ArUco → `mm_per_pixel`, stocké dans `FitMetrics` + session.
      **Marqueur non encore imprimé** — pipeline complet mais non testé sur vidéo réelle avec marqueur.
- [x] Phase 3B : `storage/` — sessions SQLite, CLI `history` + `compare`, `--save-session`.
- [x] `measure/` : bootstrap mensurations (`measure-body`) et setup vélo (`measure-setup`) — écrit les
      champs `null` de `config/rider/*.yaml` et `config/bikes/*.yaml`, avec photo/frame annotée
      (`render.py`) pour vérification visuelle. **Pas encore de tests dédiés** (`tests/` ne couvre pas
      `measure/`) — précision non validée face au mètre ruban (voir `docs/procedure-mesures.md`,
      critère de succès Phase 3A+ : ArUco ≈ ruban à ± 5 mm).
- [ ] Phase 3A+ : 2ᵉ marqueur ArUco sur le BB → mesure automatique hauteur selle, reach, stack depuis la vidéo.
- [x] Phase 3C : vue de face câblée dans `analyze --front` (fichier ou flux MJPEG ESP32-CAM), métriques
      déviation genoux + largeur coudes (`kinematics/frontal.py`, testé sur données synthétiques :
      `tests/test_frontal.py`, `test_aggregate_frontal.py`, `test_ingest.py`), schéma vélo face dans le
      rapport HTML, firmware ESP32-CAM (`firmware/`). Testé manuellement en direct (vidéo + flux réel),
      pas de jalon de précision chiffré pour l'instant (contrairement au ±2° de la Phase 1).
- [ ] Phase 4 (option) : UI Streamlit.

## Commandes
```bash
# Environnement
uv sync --all-extras

# Tests (87 tests : angles, ArUco, recommandations, storage, plan frontal, ingest MJPEG, report HTML, CLI)
python -m pytest

# Lint + format
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts

# Pipeline complet (sans calibration)
python -m bikefit video.mp4 --side right --profile triathlon_XL

# Pipeline avec calibration ArUco (marqueur 100 mm collé sur le cadre)
python -m bikefit video.mp4 --side right --marker-size 100 --save-session

# Pipeline avec vue de face (Phase 3C) — fichier ou flux MJPEG ESP32-CAM
python -m bikefit video.mp4 --side right --front video_face.mp4
python -m bikefit video.mp4 --side right --front http://192.168.1.42/stream --front-duration 60

# Historique des sessions
python -m bikefit history

# Comparer deux sessions
python -m bikefit compare 1 2

# Bootstrap mensurations depuis une photo debout
python -m bikefit measure-body photo.jpg --write

# Bootstrap setup vélo depuis une vidéo (marqueur ArUco requis)
python -m bikefit measure-setup video.mp4 --marker-size 100 --write

# Spike Phase 0
python scripts/spike_bikefit.py video.mp4 --side right
```

## Conventions de code
- Python 3.11+, type hints partout, Pydantic v2 pour toute structure de données publique.
- Tests pytest : la géométrie (angles), ArUco (frames synthétiques OpenCV) et les recommandations
  sont testées sur données synthétiques (`tests/`), jamais uniquement sur des vidéos réelles.
- Clé de cache : `md5(str(path.resolve()) + str(mtime) + str(size))[:12]` — rapide, pas de hash contenu.
- Angles vectorisés sur l'ensemble des frames (NumPy), pas de boucle frame par frame.
- Convention y-bas (OpenCV) : PMB = maximum local de `ankle_y`, détecté par `scipy.find_peaks`.
- Pas de dépendance GPU obligatoire : tout tourne sur CPU.
- `ruff check` + `ruff format --check` obligatoires en CI (lint + format séparés).

## Données runtime (~/.bikefit/)
Les données générées pendant l'utilisation vivent dans `~/.bikefit/`, hors du repo :
- `~/.bikefit/models/pose_landmarker_heavy.task` : modèle MediaPipe (téléchargé automatiquement si absent).
- `~/.bikefit/cache/landmarks/` : landmarks Parquet par vidéo (clé md5, rejoué en < 1 s).
- `~/.bikefit/sessions.db` : base SQLite des sessions (commandes `history`, `compare`).

Ces chemins sont définis dans `src/bikefit/_paths.py` (`MODELS_DIR`, `CACHE_DIR`) et correspondent
à la même racine `~/.bikefit/` que la base de sessions par défaut du CLI.

## Matériel de l'utilisateur (contexte capture)
- Tacx Neo 2 (direct drive), smartphone 1080p/30fps, vue profil = source principale.
- ESP32-CAM (basse résolution ~15-25 fps) — vue de face (Phase 3C), flux MJPEG HTTP, `--front`.
  Firmware et doc de flash dans `firmware/`. Limite matérielle : un seul client HTTP à la fois
  (`WiFiServer` basique côté firmware) — fermer tout autre lecteur du flux avant de lancer une capture.
- Le module ingest accepte fichier vidéo ET flux réseau MJPEG (`http(s)://`/`rtsp://`).
- Vidéos non versionnées (trop volumineuses) — dans `data/`, gitignorées.
- Marqueur ArUco : `DICT_4X4_50 ID=0`, 100 mm de côté, collé sur tube de selle/diagonal côté caméra.
  Fichier prêt à imprimer : `docs/aruco_marker_100mm.png`.
