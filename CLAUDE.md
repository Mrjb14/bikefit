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
- `data/` : données locales, **entièrement gitignoré**. Séparation stricte entrées/sorties :
  - `data/sources/` : vidéos et photos d'entrée (les seuls fichiers non regénérables).
  - `data/outputs/` : tout ce que le CLI produit — `reports/` (HTML), `metrics/` (JSON),
    `debug/` (vidéos `--debug-video`), `annotated/` (images de `measure-body`/`measure-setup`).

  Une analyse n'écrit **jamais** à côté de sa source : les chemins par défaut viennent de
  `REPORTS_DIR` / `ANNOTATED_DIR` (`_paths.py`), surchargeables par `BIKEFIT_OUTPUTS_DIR`.
  Attention : la clé de cache des landmarks contient le chemin résolu de la vidéo —
  déplacer un fichier source invalide son cache (renommer le `.parquet` correspondant
  dans `~/.bikefit/cache/landmarks/` évite une ré-inférence complète).

## Principes non négociables
1. **Précision avant tout** : chaque mesure est moyennée sur N cycles de pédalage,
   jamais sur une frame unique. Toute métrique porte un écart-type / indice de confiance.
   Concrètement (`kinematics/visibility.py`) : MediaPipe *infère* les points masqués
   (côté opposé à la caméra ≈ 0.2-0.3 de `visibility`). Un angle dont un sommet passe
   sous `VISIBILITY_THRESHOLD` (0.5) est mis à NaN plutôt que calculé sur des
   coordonnées devinées, et `AngleStats` porte `visibility` + `frames_kept`
   (→ `is_reliable`). Ne jamais publier une métrique sans sa fiabilité.
2. **Cache des landmarks** : la pose estimation d'une vidéo n'est calculée qu'une
   fois, résultat en Parquet dans le cache. Tout le reste du pipeline doit pouvoir
   se rejouer depuis ce cache en < 1 s.
3. **Modules découplés** : ingest → pose → calibration → kinematics → recommend → report.
   L'interface `PoseEstimator` est une abstraction (pattern Strategy) : ne jamais
   coupler le code métier à MediaPipe directement.
4. **Unités** : mm et degrés partout. Axe y image vers le bas (convention OpenCV) —
   attention aux signes dans les calculs d'angles.

## Architecture (src/bikefit/)
- `ingest/` : `VideoReader` — fichier vidéo (OpenCV), flux réseau MJPEG (`http(s)://`/`rtsp://`,
  ex. ESP32-CAM, via parsing manuel des marqueurs JPEG SOI/EOI) ou webcam locale (`webcam:<index>`,
  ex. `webcam:0`, via `cv2.VideoCapture(index)`), itérateur de frames horodatées (`TimedFrame`).
  Flux réseau et webcam (`is_live`) : timestamps sur horloge murale, pas de fps fiable côté source,
  pas de fin naturelle → capture bornée dans le temps côté CLI (`--duration`/`--front-duration`).
  `_same_source(a, b)` : compare deux sources (fichier résolu, ou chaîne exacte pour flux/webcam) —
  utilisé par le CLI pour interdire de lancer les vues profil et face sur la même caméra.
- `pose/` : protocole `PoseEstimator` + `MediaPipePoseEstimator` (Tasks API, `pose_landmarker_heavy`) + `LandmarkCache` (Parquet).
- `kinematics/` : `filter_landmarks` (Savitzky-Golay ~6 Hz), `detect_cycles` (extrema cheville y,
  renvoie des **`frame_idx`** — pas des positions de tableau, cf. découpage `--start/--end` — et
  rejette via `_filter_implausible_cycles` tout cycle dont la durée s'écarte de plus de 35% de la
  médiane, à partir de 3 cycles détectés),
  `compute_angles` (genou, hanche, tronc, coude, cheville — vectorisé NumPy), `aggregate_cycles` → `FitMetrics`.
  `visibility.py` : `VISIBILITY_THRESHOLD` (0.5), `reliability_mask`, `weakest_visibility` — filtre les
  points *inférés* par MediaPipe (cf. principe n°1). `compute_angles` / `compute_frontal_metrics`
  émettent aussi une colonne `<metrique>_vis` par frame, consommée par l'agrégation et le HUD de debug.
  `frontal.py` (Phase 3C, vue de face) : `compute_frontal_metrics` (déviation angulaire genou vs droite
  hanche-cheville, largeur coudes px) + `aggregate_frontal_cycles` → champs optionnels de `FitMetrics`.
  `AngleStats` porte `visibility` / `frames_kept` (→ `is_reliable`), optionnels pour rester
  rétrocompatible avec les sessions déjà enregistrées.
  `FitMetrics` porte des champs optionnels `calibration: CalibrationResult | None` et plan frontal
  (`genou_deviation_gauche/droit`, `largeur_coudes`).
- `calibration/` : `detect_marker_size_px` (frame unique) + `calibrate_from_video` (vidéo entière,
  médiane sur frames échantillonnées) → `CalibrationResult(mm_per_pixel, n_detected, n_sampled, is_reliable)`.
  Dictionnaire ArUco : `DICT_4X4_50`. Marqueur ID=0.
- `equipment/` : modèles Pydantic `BikeProfile` / `BikeSetup` / `PlagesReglage`, `load_bike_profile(name)`.
- `recommend/` : `run(metrics, bike, profile)` → `FitReport` + `Recommendation` (règles explicites, pas de ML).
  Cibles dans `config/targets_tt.yaml` (profils : `clm_court`, `clm_long`, `triathlon_XL`).
  Coefficients de sensibilité empiriques (°/mm) — seront affinés par calibration ArUco (Phase 3A+).
  6 réglages : `hauteur_selle`, `recul_selle`, `hauteur_prolongateurs`, `longueur_prolongateurs`,
  `cale_chaussure` (sagittal) et `largeur_pads` (frontal). Ce dernier compare la largeur de coudes
  à la largeur d'épaules (`largeur_coudes_ratio` dans targets) : il reste **silencieux** sans
  calibration ArUco, sans mesure fiable, ou sans `largeur_epaules` renseignée.
- `report/` : `render_html(metrics, report, angle_df, video_path, output_path)` — Jinja2 + Plotly CDN.
  Dark theme, 5 graphes d'angles empilés, schéma SVG du vélo de profil + schéma vue de face (Phase 3C,
  roue en tranche, cintre symétrique) avec annotations optionnelles si métriques plan frontal présentes.
  Section « Plan frontal » et encart « peu fiable » (visibilité + % de frames) sur toute carte dont
  `is_reliable` est faux.
- `debug/` : `render_debug_video_from_landmarks` (fichier : rejeu + superposition des métriques déjà
  lissées) et `render_debug_video_live` (flux/webcam : écriture pendant l'unique passage, métriques
  brutes par frame). Le squelette **estompe les segments inférés** (trait fin gris, cercle creux).
  `DebugVideoWriter(measure_fps=True)` pour les sources live : le fps nominal du capteur n'est pas
  tenu quand l'inférence tourne en ligne (30 déclarés → ~20 réels = vidéo 1,5× trop rapide), donc le
  débit réel est mesuré sur les 2 premières secondes avant d'ouvrir le fichier.
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
- `debug/` : `DebugVideoWriter` — vidéo MP4 annotée (squelette `kinematics.landmarks.SKELETON_EDGES`
  + HUD de métriques) pour debugger la détection, quelle que soit la source (fichier, flux, webcam).
  Deux stratégies : `render_debug_video_from_landmarks` (fichier — rejoue le fichier une 2ᵉ fois,
  décodage seul, superpose les angles lissés post-filtre, déjà ceux utilisés pour les recommandations)
  et `render_debug_video_live` (flux/webcam — un flux ne se rejoue pas, écrit pendant l'unique passage
  de capture via le hook `on_frame` de `pose.compute_landmarks`, angles bruts non lissés par frame).
  CLI : `--debug-video`/`--front-debug-video` sur `analyze`.
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
- [x] Vue de profil en flux réseau/webcam (comme la vue de face) + garde anti-double-source
      (`--front` ne peut pas pointer vers la même caméra/fichier que la vidéo de profil).
- [x] `debug/` : vidéo de debug annotée (squelette + HUD d'angles) pour `analyze --debug-video` et
      `--front-debug-video`, sur fichier (angles lissés, post-pipeline) ou flux/webcam (angles bruts,
      écrits pendant la capture). Testé via un faux `PoseEstimator` (pas de MediaPipe dans les tests) :
      `tests/test_debug_video.py`, `TestAnalyzeDebugVideo` dans `tests/test_cli.py`.
- [x] Fiabilité des mesures (`kinematics/visibility.py`) : les landmarks inférés par MediaPipe sont
      exclus des angles, et chaque `AngleStats` porte sa visibilité + son taux de frames exploitées.
      Mesuré sur capture réelle : vue de profil inchangée (côté filmé à 0.94-1.00), mais 60 des
      63 cycles de la **vue de face** reposaient sur des genoux inférés (genou gauche à 0.32) — la
      déviation du genou droit passait de +2.6° à -0.3°, soit un changement de signe.
      **Conséquence matérielle** : les déviations de genoux vue de face ne sont pas exploitables avec
      le placement actuel de l'ESP32-CAM ; à revoir (hauteur/éclairage) avant d'utiliser cette métrique.
- [x] Robustesse des cycles : `detect_cycles` autorise jusqu'à 152 rpm (`min_dist = fps * 0.4`), ce qui
      laisse passer de faux PMB. Sur une capture réelle, 1 cycle parasite de 0.39 s (vs 0.79 s médian)
      faisait passer l'écart-type de `genou_extension_pmb` de 2.2° à 9.8°. `_filter_implausible_cycles`
      rejette désormais tout cycle dont la durée s'écarte de plus de 35% de la médiane (à partir de
      3 cycles détectés, sous ce seuil la médiane n'est pas fiable).
- [ ] Phase 4 (option) : UI Streamlit.

## Commandes
```bash
# Environnement
uv sync --all-extras

# Tests (209 tests : angles, filtre SG, cycles (+ plausibilité), visibilité/fiabilité, ArUco, recommandations,
# storage, plan frontal, ingest MJPEG/webcam, measure/, report HTML,
# CLI (dont pipeline complet + vidéo de debug via faux PoseEstimator))
python -m pytest

# Types (obligatoire en CI, au même titre que ruff)
uv run mypy src

# Lint + format
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts

# Pipeline complet (sans calibration)
# Le rapport atterrit dans data/outputs/reports/<video>_<horodatage>.html
python -m bikefit data/sources/video.mp4 --side right --profile triathlon_XL

# Découpage : ignorer la montée sur le vélo et la descente (indispensable en pratique,
# ces phases hors pédalage faussent les moyennes). Appliqué après le cache -> rejeu instantané.
python -m bikefit data/sources/video.mp4 --side right --start 20 --end 70

# Pipeline avec calibration ArUco (marqueur 100 mm collé sur le cadre)
python -m bikefit data/sources/video.mp4 --side right --marker-size 100 --save-session

# Pipeline avec vue de face (Phase 3C) — fichier ou flux MJPEG ESP32-CAM
python -m bikefit data/sources/video.mp4 --side right --front data/sources/video_face.mp4 \
  --start 20 --end 70 --front-start 20 --front-end 65
python -m bikefit data/sources/video.mp4 --side right --front http://192.168.1.42/stream --front-duration 60

# Vue de profil et vue de face toutes les deux en webcam (deux périphériques distincts)
python -m bikefit webcam:0 --side right --duration 60 --front webcam:1 --front-duration 60
# webcam:0 des deux côtés -> rejeté (Erreur : --front pointe vers la même source ...)

# Vidéo de debug (squelette + angles superposés) — fichier, flux ou webcam indifféremment
python -m bikefit video.mp4 --side right --debug-video debug.mp4
python -m bikefit webcam:0 --side right --duration 30 --debug-video debug_live.mp4
python -m bikefit video.mp4 --side right --front webcam:0 --front-debug-video debug_face.mp4

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
  Vaut aussi pour les `world_landmarks` 3D de `measure/body.py` (une cheville a un y *plus grand*
  qu'une hanche) — l'inverser produit des longueurs négatives, cf. `tests/test_measure_body.py`.
- Pas de dépendance GPU obligatoire : tout tourne sur CPU.
- `ruff check` + `ruff format --check` + `mypy src` obligatoires en CI (3 étapes séparées).
- CI : matrice `ubuntu` (3.11, 3.12) **et `windows`** — plateforme de dev principale, seule à
  révéler les bugs d'encodage console (cp1252) et de séparateurs de chemin. Seuil de couverture
  `--cov-fail-under=65`.

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
- Le module ingest accepte fichier vidéo, flux réseau MJPEG (`http(s)://`/`rtsp://`) ET webcam locale
  (`webcam:<index>`) — vue de profil et vue de face acceptent les trois indifféremment, avec une
  vérification au lancement qu'elles ne pointent pas vers la même source.
- Vidéos non versionnées (trop volumineuses) — dans `data/sources/`, gitignorées.
- Marqueur ArUco : `DICT_4X4_50 ID=0`, 100 mm de côté, collé sur tube de selle/diagonal côté caméra.
  Fichier prêt à imprimer : `docs/aruco_marker_100mm.png`.
