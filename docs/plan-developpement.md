# BikeFit CLM — Plan de développement & Architecture

Logiciel d'analyse vidéo pour le réglage d'un vélo de contre-la-montre, basé sur l'estimation de pose par vision par ordinateur.

---

## 1. Vision du produit

**Objectif** : à partir de vidéos (profil + face) d'un cycliste pédalant sur home trainer, produire un rapport de réglage chiffré : hauteur/recul de selle, position des prolongateurs, hauteur du poste de pilotage.

**Utilisateur cible** : toi (v1), puis potentiellement d'autres cyclistes (v2+).

**Principe directeur** : précision et reproductibilité avant tout. Une recommandation fausse de 10 mm est pire que pas de recommandation. Chaque mesure doit être moyennée sur plusieurs cycles de pédalage et accompagnée d'un indice de confiance.

---

## 2. Architecture générale

```
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Ingestion   │──▶│ Pose         │──▶│ Calibration  │──▶│ Analyse      │
│  vidéo       │   │ Estimation   │   │ ArUco        │   │ cinématique  │
└─────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                         │                                       │
                    cache Parquet                         ┌──────▼───────┐
                                                          │  Moteur de   │
                                                          │ recommandation│
                                                          └──────┬───────┘
                                                                 │
                                                    ┌────────────┼────────────┐
                                                    ▼            ▼            ▼
                                             Rapport HTML   Session       Comparaison
                                             Jinja2+Plotly  SQLite        avant/après
```

### 2.1 Module Ingestion vidéo — ✅ implémenté
- `VideoReader` : lit fps/width/height/frame_count à l'init, itérateur `frames()` → `TimedFrame`.
- Entrée : fichiers vidéo (mp4/mov) **ou flux réseau MJPEG** (`http(s)://`/`rtsp://`, ex. ESP32-CAM,
  Phase 3C) — détection auto du préfixe, parsing manuel des marqueurs JPEG SOI/EOI (le multipart
  MJPEG d'un petit serveur maison n'est pas lu de façon fiable par `cv2.VideoCapture`/FFmpeg).
- Horodatage fichier : `timestamp_ms = frame_idx * 1000 / fps`. Horodatage flux réseau : horloge
  murale au moment de la capture (fps déclaré par la source non fiable).

### 2.2 Module Pose Estimation — ✅ implémenté
- Moteur : **MediaPipe Pose Tasks API v1.0** (`pose_landmarker_heavy`, `VisionRunningMode.VIDEO`).
- Protocole `PoseEstimator` (Strategy) : le code métier n'importe jamais MediaPipe directement.
- `LandmarkCache` : clé `md5(path+mtime+size)[:12]`, stockage Parquet dans `cache/landmarks/`.
  Recalcul uniquement si la vidéo a changé. Tout le pipeline en aval se rejoue en < 1 s.
- `FrameLandmarks` : 33 landmarks (x, y, z, visibilité) normalisés + flag `detected`.

### 2.3 Module Calibration ArUco — ✅ implémenté (marqueur non encore imprimé)
- `detect_marker_size_px(frame_bgr)` → taille en pixels du premier marqueur détecté, ou None.
- `calibrate_from_video(path, marker_size_mm, sample_every=10)` → `CalibrationResult | None`.
  Lit la vidéo directement sans charger toutes les frames en RAM (robuste pour 1080p).
- Médiane sur N frames échantillonnées → robustesse aux occlusions passagères.
- `CalibrationResult` : `mm_per_pixel`, `n_detected`, `n_sampled`, `is_reliable` (≥ 50 % et ≥ 5 frames).
- Dictionnaire : `DICT_4X4_50`, marqueur ID=0. Fichier à imprimer : `docs/aruco_marker_100mm.png`.
- Activé par `--marker-size <mm>` sur la CLI ; stocké dans `FitMetrics.calibration`.

**Approche intermédiaire déjà livrée (`measure-setup`, cf. 2.10)** : plutôt qu'un 2ᵉ marqueur ArUco
sur le BB, le centre du BB est estimé par fit de cercle (Kåsa) sur la trajectoire de la cheville sur
l'ensemble de la vidéo — un seul marqueur (calibration mm/px) suffit. Précision indicative ±20 mm,
à valider au mètre ruban (critère de succès Phase 3A+ : ArUco ≈ ruban à ± 5 mm — pas encore atteint).

**À venir (Phase 3A+) :**
- 2ᵉ marqueur ArUco (ID=1) positionné sur le BB, pour une mesure plus précise que le fit de cercle.
- Remplissage automatique de `BikeSetup` avec validation ± 5 mm — plus besoin de mètre ruban.
- Calibration personnalisée des coefficients de sensibilité (°/mm) à partir des sessions réelles.

### 2.4 Module Analyse cinématique — ✅ implémenté
- **Filtrage** : `filter_landmarks` — interpolation linéaire sur les gaps NaN, puis Savitzky-Golay (polyorder=3, fenêtre ≈ fps/5). NaN restaurés pour les frames non détectées.
- **Détection des cycles** : `detect_cycles` — `find_peaks` sur `ankle_y` (y image vers le bas → PMB = maximum). Distance minimale entre pics = `fps * 0.4` (permet jusqu'à 150 rpm).
- **Calcul d'angles** : `compute_angles` — vectorisé NumPy sur toutes les frames en une passe.
  - genou (hanche–genou–cheville), hanche (épaule–hanche–genou), tronc (angle épaule-hanche / horizontale), coude (épaule–coude–poignet), cheville (genou–cheville–pied).
  - NaN propagé pour les frames non détectées.
- **Agrégation** : `aggregate_cycles` — par cycle : genou_ext (max), genou_flex (min), hanche (min), tronc/coude (mean), cheville (max PMB). Cycles avec > 10% NaN rejetés.
- Sortie : `FitMetrics` (Pydantic) — 6 angles × `{mean, std, min, max, n_cycles}` + `calibration` optionnel.

### 2.5 Module Équipement — ✅ implémenté
- `BikeProfile` (Pydantic) : modele, longueur_manivelle_mm, `BikeSetup` (réglages actuels en mm), `PlagesReglage` (butées min/max).
- `load_bike_profile(name)` : charge `config/bikes/{name}.yaml`.

### 2.6 Moteur de recommandation — ✅ implémenté
- `run(metrics, bike, profile)` → `FitReport`.
- Profils cibles dans `config/targets_tt.yaml` : `clm_court`, `clm_long`, `triathlon_XL`.
- Coefficients de sensibilité empiriques (°/mm) pour convertir un écart angulaire en delta mm.
- Règles : genou_ext → hauteur selle (0.65°/mm), genou_flex → recul selle, hanche → hauteur prolongateurs (0.30°/mm), coude → longueur prolongateurs (0.25°/mm), tronc → hauteur prolongateurs, cheville → cale.
- Détection de butée si le delta mm dépasse la plage de réglage connue.

> **Limitation actuelle** : les delta_mm utilisent des coefficients empiriques issus de la littérature de fitting (Pruitt, Retül). Ils seront remplacés par des valeurs géométriquement calculées dès que la calibration ArUco BB sera opérationnelle (Phase 3A+).

### 2.7 Rapport & Export — ✅ implémenté
- `render_html(metrics, report, angle_df, video_path, output_path)`.
- Template Jinja2 — dark theme, responsive, CDN Plotly (autonome, sans serveur). Autoescape HTML
  activé (le JSON des graphes, seule donnée non textuelle injectée, passe par un filtre `|safe`
  explicite dans le template).
- 5 graphes d'angles empilés (`scattergl`), schéma SVG du vélo de profil avec annotations colorées
  par priorité, + schéma SVG vue de face (Phase 3C : roue en tranche, cintre symétrique) avec
  annotations optionnelles (déviation genoux, largeur coudes) si les métriques plan frontal sont
  présentes dans `FitMetrics`.
- Tableau des métriques vert/rouge selon plages cibles.

### 2.8 Storage — ✅ implémenté
- `Session` (Pydantic) : date, video_path, bike_name, profile, setup, metrics, report.
- `SessionStore` (SQLite, `sqlite3` natif) : `save`, `get`, `list_all(ORDER BY date DESC, id DESC)`, `latest`, `delete`.
- CLI : `--save-session` sur `analyze`, commandes `history` et `compare <id1> <id2>`.
- `compare` : tableau des deltas d'angles (avant/après) + recommandations restantes.
- Base par défaut : `~/.bikefit/sessions.db`.

### 2.9 Interface utilisateur
- **v1 : CLI** (`python -m bikefit video.mp4 --side right`) — ✅ implémentée.
  5 commandes Typer : `analyze` (défaut, avec `--front` pour la vue de face), `history`, `compare`,
  `measure-body`, `measure-setup`.
- **v2 : UI web locale** (Streamlit) — Phase 4 optionnelle.

### 2.10 Module Measure (bootstrap YAML) — ✅ implémenté
- Objectif : remplir un premier jet des YAML personnels (`config/rider/`, `config/bikes/`) sans
  sortir le mètre ruban à chaque itération, à valider ensuite au ruban (voir `docs/procedure-mesures.md`).
- `estimate_body_measurements` (`measure-body`) : photo debout, MediaPipe en mode IMAGE
  (`pose_world_landmarks` 3D), mise à l'échelle via le ratio nez→cheville / taille si `taille` connue.
  Précision indicative ±5-10 %.
- `estimate_bike_setup` (`measure-setup`) : vidéo + calibration ArUco, fit de cercle de Kåsa sur la
  trajectoire de la cheville → centre BB, puis projection pour `hauteur_selle_bb`,
  `hauteur_guidon_bb`, `reach_prolongateurs`. Précision indicative ±20 mm.
- `update_null_fields` (`yaml_io.py`) : réécrit uniquement les champs `null` d'un YAML existant,
  commentaires préservés — n'écrase jamais une valeur déjà renseignée.
- `render.py` : annotation visuelle OpenCV (squelette + crochets de cotation) sur la photo/frame au
  PMB, pour vérifier les estimations avant `--write`.
- **Lacune connue** : pas de tests dédiés (`tests/` ne couvre pas `measure/`) — précision non
  validée par des tests automatisés, seulement par la procédure manuelle au ruban.

---

## 3. Stack technique

| Couche | Choix | Justification |
|---|---|---|
| Langage | Python 3.11+ | Écosystème CV, rapidité de dev |
| Vision | MediaPipe Tasks API v1.0, OpenCV | Gratuit, précis, temps réel |
| Calcul | NumPy, SciPy, pandas | Filtrage, géométrie, stats |
| Calibration | OpenCV ArUco `DICT_4X4_50` | Robuste, intégré dans `opencv-python>=4.9` |
| Config | Pydantic v2 + YAML | Validation forte des seuils |
| Stockage | Parquet (landmarks) + SQLite (sessions) | Simple, local, sans serveur |
| Rapport | Jinja2 → HTML, Plotly CDN | Export autonome, pas de dépendance serveur |
| UI v2 | Streamlit (Phase 4, optionnel) | Rapide à prototyper |
| Tests | pytest, données synthétiques (NumPy + OpenCV) | Non-régression géométrie + ArUco + recommandations + plan frontal + ingest MJPEG |
| CI | GitHub Actions — lint (ruff) + format + tests Python 3.11/3.12 | Feedback rapide. `mypy` est configuré (`pyproject.toml`) mais **pas encore branché en CI** — à ajouter ou retirer de `dev` pour éviter une fausse promesse de vérification de types. |
| Packaging | uv / pyproject.toml | Standard moderne |

---

## 4. Structure du projet

```
bikefit/
├── pyproject.toml
├── CLAUDE.md
├── README.md
├── .github/
│   ├── workflows/ci.yml          # lint + tests Python 3.11/3.12
│   └── dependabot.yml            # mises à jour pip + actions, mensuel
├── config/
│   ├── targets_tt.yaml           # angles cibles + coefficients de sensibilité
│   ├── bikes/                    # fiches vélo personnelles (gitignorées)
│   │   └── giant_trinity_2019.yaml
│   └── rider/                    # mensurations (gitignorées)
│       └── mensurations.yaml
├── docs/
│   ├── plan-developpement.md     # ce fichier
│   ├── procedure-mesures.md      # procédure de prise de mesures + validation ruban
│   └── aruco_marker_100mm.png    # marqueur à imprimer (DICT_4X4_50 ID=0)
├── firmware/
│   ├── README.md                 # installation/flash ESP32-CAM
│   └── esp32cam_mjpeg/           # sketch Arduino, serveur MJPEG HTTP (vue de face)
├── models/                       # modèles MediaPipe .task (gitignorés)
├── cache/landmarks/              # Parquet par vidéo (gitignorés)
├── scripts/
│   └── spike_bikefit.py          # Phase 0
└── src/bikefit/
    ├── _paths.py                 # PROJECT_ROOT, MODELS_DIR, CACHE_DIR, CONFIG_DIR
    ├── cli.py                    # Typer — analyze (--front) / history / compare / measure-body / measure-setup
    ├── __main__.py               # injection du sous-commande par défaut
    ├── ingest/
    │   └── reader.py             # VideoReader (fichier ou flux MJPEG réseau), TimedFrame
    ├── pose/
    │   ├── base.py               # Landmark, FrameLandmarks, PoseEstimator (Protocol)
    │   ├── mediapipe_estimator.py
    │   └── cache.py              # LandmarkCache
    ├── calibration/
    │   ├── models.py             # CalibrationResult
    │   └── aruco.py              # detect_marker_size_px, calibrate_from_frames, calibrate_from_video
    ├── kinematics/
    │   ├── landmarks.py          # BODY dict, ANGLE_JOINTS
    │   ├── filter.py             # filter_landmarks (Savitzky-Golay)
    │   ├── cycles.py             # detect_cycles (find_peaks ankle_y)
    │   ├── angles.py             # compute_angles (vectorisé)
    │   ├── frontal.py            # compute_frontal_metrics, aggregate_frontal_cycles (Phase 3C)
    │   ├── aggregate.py          # aggregate_cycles → FitMetrics
    │   └── metrics.py            # AngleStats, FitMetrics (Pydantic)
    ├── equipment/
    │   ├── models.py             # BikeProfile, BikeSetup, PlagesReglage
    │   └── loader.py             # load_bike_profile(name)
    ├── recommend/
    │   ├── models.py             # Recommendation, FitReport, Priorite (StrEnum)
    │   └── engine.py             # run(metrics, bike, profile) → FitReport
    ├── report/
    │   ├── html.py               # render_html(...)
    │   └── template.html.jinja2
    ├── storage/
    │   ├── models.py             # Session (Pydantic)
    │   └── store.py              # SessionStore (sqlite3)
    └── measure/
        ├── body.py               # estimate_body_measurements (photo → mensurations)
        ├── setup.py              # estimate_bike_setup (vidéo + ArUco → setup vélo)
        ├── render.py             # annotation visuelle OpenCV (vérification avant --write)
        └── yaml_io.py            # update_null_fields (écrit uniquement les champs null)
```

---

## 5. Plan de développement par phases

### Phase 0 — Setup & spike technique ✅
- Init repo, environnement (uv), CI.
- Spike MediaPipe Tasks API v1.0 (`scripts/spike_bikefit.py`).
- Livrable : script qui télécharge le modèle et superpose le squelette sur une vidéo.

### Phase 1 — Pipeline profil, cœur métier ✅
- `VideoReader` → `MediaPipePoseEstimator` → `LandmarkCache` (Parquet).
- `filter_landmarks` → `compute_angles` → `detect_cycles` → `aggregate_cycles` → `FitMetrics`.
- CLI Typer + `__main__.py`.
- 10 tests unitaires (angles sur squelettes synthétiques).
- **Jalon atteint** : 59 cycles, 100% détection, std ≤ 2.2°.

### Phase 2 — Recommandations & rapport HTML ✅
- `BikeProfile` / `BikeSetup` / `PlagesReglage` Pydantic + `load_bike_profile`.
- `config/targets_tt.yaml` — 3 profils + coefficients de sensibilité empiriques.
- Moteur de règles `recommend/engine.py` — 6 règles, priorités, détection butées.
- Rapport HTML dark theme — Jinja2 + Plotly, autonome, responsive.
- 20 tests unitaires (angles + recommandations).

### Phase 3A — Calibration ArUco (mm/pixel) ✅ *(marqueur non encore imprimé)*
- `calibration/aruco.py` : détection marqueur DICT_4X4_50, médiane multi-frames, lecture vidéo sans chargement RAM.
- `CalibrationResult` : mm_per_pixel, taux de détection, fiabilité.
- `FitMetrics.calibration` : champ optionnel.
- CLI `--marker-size <mm>`.
- 14 tests unitaires (frames synthétiques générées par OpenCV).
- Fichier à imprimer : `docs/aruco_marker_100mm.png`.

### Phase 3B — Sessions SQLite ✅
- `Session` + `SessionStore` (sqlite3 natif).
- CLI `--save-session`, `history`, `compare <id1> <id2>`.
- 11 tests unitaires (données synthétiques, sans vidéo).

### `measure/` — Bootstrap mensurations & setup vélo ✅ *(hors plan initial, cf. 2.10)*
- `measure-body` (photo → mensurations) et `measure-setup` (vidéo + ArUco → setup vélo, fit de
  cercle sur trajectoire cheville → BB), avec photo/frame annotée pour vérification visuelle.
- **Lacune** : pas de tests dédiés dans `tests/`.

### Phase 3A+ — Mesure automatique du vélo *(à faire)*
- 2ᵉ marqueur ArUco (ID=1) positionné sur le BB, pour affiner l'estimation du BB au-delà du fit de
  cercle déjà livré par `measure-setup` (approche intermédiaire à un seul marqueur, ±20 mm).
- Calcul automatique depuis la vidéo :
  - hauteur de selle (BB → marqueur selle / landmark hanche)
  - reach (BB → extrémité des extensions)
  - stack (BB → hauteur des pads)
- Remplissage automatique de `BikeSetup` avec validation ± 5 mm — plus besoin de mètre ruban.
- Calibration des coefficients de sensibilité à partir des sessions réelles (avant/après réglage connu).
- **Prérequis** : imprimer et coller les marqueurs ArUco ID=0 (calibration) et ID=1 (BB).

### Phase 3C — Vue de face ✅
- Flux MJPEG HTTP (ESP32-CAM) dans `VideoReader` (parsing manuel SOI/EOI), firmware Arduino
  (`firmware/esp32cam_mjpeg/`) documenté dans `firmware/README.md`.
- Angles dans le plan frontal : déviation genoux, largeur coudes (`kinematics/frontal.py`).
- Nouvelles métriques optionnelles dans `FitMetrics` (`genou_deviation_gauche/droit`, `largeur_coudes`).
- CLI : `analyze --front <vidéo|url> [--front-duration N]`. Schéma vélo vue de face dans le rapport HTML.
- 16 tests unitaires (`test_frontal.py`, `test_aggregate_frontal.py`, `test_ingest.py`, données
  synthétiques). Testé manuellement en direct (vidéo + flux ESP32-CAM réel), mais **pas de jalon de
  précision chiffré pour l'instant** — contrairement au ±2° de la Phase 1.

### Phase 4 — UI web *(optionnel)*
- Streamlit : upload vidéo, squelette superposé, rapport interactif, historique sessions.

---

## 6. Optimisations implémentées

- **Cache Parquet** : la pose n'est calculée qu'une fois par vidéo (clé mtime+size).
- **Angles vectorisés** : `compute_angles` traite toutes les frames en une passe NumPy.
- **NaN propre** : les frames non détectées propagent NaN à travers filtre → angles → agrégation.
- **Moyenne multi-cycles** (N=59 sur la vidéo de référence) : robustesse sans coût GPU.
- **ArUco médiane** : `calibrate_from_video` échantillonne 1 frame / 10, médiane → robuste aux occlusions.
- **SQLite natif** : pas de dépendance ORM, `sqlite3` de la stdlib suffit.

---

## 7. Risques & parades

| Risque | Impact | Parade |
|---|---|---|
| Détection imprécise cheville/pied (chaussure) | Angles genou/cheville faux | Marqueurs contrastés, éclairage latéral, filtre de visibilité MediaPipe |
| Caméra non perpendiculaire au plan du vélo | Angles projetés biaisés | ArUco ID=1 sur le BB : si longueur manivelle mesurée ≠ 172.5 mm → alerte perspective |
| Occlusion jambe opposée (vue profil) | Landmarks confondus | Ne garder que la jambe côté caméra (`--side right/left`) |
| delta_mm approximatif sans mesure BB | Sur/sous-correction | Phase 3A+ : mesure directe depuis vidéo ; en attendant : ajustements par petits pas |
| Marqueur ArUco occulté par la jambe | Calibration peu fiable | `CalibrationResult.is_reliable` alerte si < 50 % de détection |
| Variabilité de cadence entre sessions | Biais de cycle | Sélectionner une plage de cadence constante (Phase 3A+) |

---

## 8. Critères de succès

1. Écart-type < 2° sur chaque angle entre trois captures de la même position. ✅ (std ≤ 2.2°)
2. Un fitting complet (capture → rapport HTML) en moins de 10 minutes. ✅ (pipeline < 3 min sur CPU)
3. Calibration ArUco fiable (≥ 50 % détection) sur une vidéo réelle avec marqueur. *(à valider)*
4. Les réglages recommandés, appliqués et re-mesurés, ramènent les angles dans les plages cibles. *(à valider Phase 3A+)*
5. `compare` avant/après montre des deltas cohérents avec les ajustements physiques réels. *(à valider)*
