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
- `LandmarkCache` : clé `md5(path+mtime+size)[:12]`, stockage Parquet dans `~/.bikefit/cache/landmarks/`
  (les données runtime — modèle, cache, `sessions.db` — vivent hors du dépôt).
  ⚠️ La clé contient le chemin **résolu** : déplacer une vidéo source invalide son cache.
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

**Approche intermédiaire (`measure-setup`, cf. 2.10)** : à défaut de marqueur BB détecté, le centre du
BB est estimé par fit de cercle (Kåsa) sur la trajectoire de la cheville sur l'ensemble de la vidéo —
un seul marqueur (calibration mm/px) suffit. Précision indicative ±20 mm.

**Phase 3A+ — mesure automatique du vélo (2ᵉ marqueur BB) :**
- `detect_marker_size_px`/`detect_marker_center_px` (`calibration/aruco.py`) filtrent désormais par
  `marker_id` — indispensable dès que 2 marqueurs cohabitent dans le champ (ID=0 calibration, ID=1 BB),
  sans quoi le premier marqueur renvoyé par OpenCV est arbitraire. `calibrate_from_video` calibre
  explicitement sur l'ID=0 (n'est plus contaminé par l'ID=1 s'il est visible).
- `locate_marker_in_video(video_path, marker_id, sample_every=10)` → `MarkerPosition | None` : position
  pixel médiane d'un marqueur fixe (robuste aux occlusions passagères, ex. pied devant le BB).
- `measure-setup` (CLI) cherche automatiquement le marqueur BB (ID=1) ; s'il est détecté de façon fiable
  (≥ 50 % des frames échantillonnées, ≥ 5 détections), `estimate_bike_setup` l'utilise directement comme
  BB (±5 mm annoncé) au lieu du fit de cercle (±20 mm) — repli automatique et transparent si absent.
  Le rayon cheville (crank + hauteur chaussure) est alors calculé par rapport à ce BB réel, pas au
  centre du cercle de Kåsa.
- Testé sur frames/vidéos ArUco synthétiques (`tests/test_aruco.py`, `tests/test_measure_setup.py`) —
  **critère de succès ArUco ≈ ruban à ± 5 mm pas encore validé sur capture réelle** (marqueur ID=1
  physiquement collé, validation au mètre ruban à faire).
- **Reste à faire** : calibration personnalisée des coefficients de sensibilité (°/mm) à partir de
  sessions réelles avant/après réglage — bloqué tant que `setup_actuel` n'a pas de première mesure de
  référence (voir doc procédure de mesures).

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
- **v2 : UI web locale** (Streamlit, `src/bikefit/ui/`) — ✅ MVP implémenté (`analyze` uniquement,
  cf. section 5).

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
| UI v2 | Streamlit (Phase 4, MVP fait) | Rapide à prototyper |
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
│   ├── aruco_marker_100mm.png    # marqueur calibration à imprimer (DICT_4X4_50 ID=0)
│   └── aruco_marker_id1_100mm.png # marqueur BB à imprimer (DICT_4X4_50 ID=1, Phase 3A+)
├── firmware/
│   ├── README.md                 # installation/flash ESP32-CAM
│   └── esp32cam_mjpeg/           # sketch Arduino, serveur MJPEG HTTP (vue de face)
├── data/                         # données locales, entièrement gitignoré
│   ├── sources/                  # vidéos et photos d'entrée
│   └── outputs/                  # reports/ · metrics/ · debug/ · annotated/
├── scripts/
│   └── spike_bikefit.py          # Phase 0
└── src/bikefit/
    ├── _paths.py                 # PROJECT_ROOT, MODELS_DIR, CACHE_DIR, CONFIG_DIR,
    │                             #   REPORTS_DIR, ANNOTATED_DIR (BIKEFIT_OUTPUTS_DIR)
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
    ├── measure/
    │   ├── body.py               # estimate_body_measurements (photo → mensurations)
    │   ├── setup.py              # estimate_bike_setup (vidéo + ArUco → setup vélo)
    │   ├── render.py             # annotation visuelle OpenCV (vérification avant --write)
    │   └── yaml_io.py            # update_null_fields (écrit uniquement les champs null)
    ├── debug/
    │   └── video.py              # DebugVideoWriter, render_debug_video_from_landmarks/_live
    └── ui/                       # Phase 4 (MVP) — app Streamlit
        ├── app.py                # formulaire (upload, config) → cli.analyze → rapport affiché
        └── helpers.py            # logique pure (parsing, listing configs, save upload), testée
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

### Phase 3A+ — Mesure automatique du vélo *(BB par marqueur livré, validation réelle à faire)*
- 2ᵉ marqueur ArUco (ID=1) sur le BB : `locate_marker_in_video` le localise (position pixel médiane,
  robuste aux occlusions), `estimate_bike_setup` l'utilise comme centre BB à la place du fit de cercle
  quand il est détecté de façon fiable (repli automatique sinon — approche intermédiaire à un seul
  marqueur, ±20 mm, conservée comme filet de sécurité).
- Calcul automatique depuis la vidéo, relatif à ce BB :
  - hauteur de selle (BB → hanche au PMB)
  - reach (BB → poignet médian, horizontal)
  - hauteur guidon / stack (BB → poignet médian, vertical)
- CLI `measure-setup` : détection + bascule automatiques, message de précision (±5 mm marqueur /
  ±20 mm cercle) affiché à l'utilisateur.
- Testé sur frames/vidéos ArUco synthétiques uniquement pour l'instant.
- **Reste à faire** :
  - valider sur vidéo réelle (marqueur ID=1 collé) que ArUco ≈ ruban à ± 5 mm (critère de succès non
    encore vérifié en conditions réelles) ;
  - calibration des coefficients de sensibilité à partir des sessions réelles (avant/après réglage
    connu) — bloqué tant qu'une première mesure de référence n'existe pas.

### Phase 3C — Vue de face ✅
- Flux MJPEG HTTP (ESP32-CAM) dans `VideoReader` (parsing manuel SOI/EOI), firmware Arduino
  (`firmware/esp32cam_mjpeg/`) documenté dans `firmware/README.md`.
- Angles dans le plan frontal : déviation genoux, largeur coudes (`kinematics/frontal.py`).
- Nouvelles métriques optionnelles dans `FitMetrics` (`genou_deviation_gauche/droit`, `largeur_coudes`).
- CLI : `analyze --front <vidéo|url> [--front-duration N]`. Schéma vélo vue de face dans le rapport HTML.
- 16 tests unitaires (`test_frontal.py`, `test_aggregate_frontal.py`, `test_ingest.py`, données
  synthétiques). Testé manuellement en direct (vidéo + flux ESP32-CAM réel), mais **pas de jalon de
  précision chiffré pour l'instant** — contrairement au ±2° de la Phase 1.

### Fiabilité des mesures ✅ *(hors plan initial — mise en œuvre du principe n°1)*
- `kinematics/visibility.py` : seuil `VISIBILITY_THRESHOLD` (0.5), `reliability_mask`,
  `weakest_visibility`. Un angle dont un sommet est *inféré* par MediaPipe est mis à NaN.
- `AngleStats` porte `visibility` + `frames_kept` (→ `is_reliable`), optionnels donc
  rétrocompatibles avec les sessions déjà enregistrées.
- Restitution : avertissement CLI, encart « peu fiable » et section Plan frontal dans le rapport
  HTML, segments estompés dans les vidéos de debug.
- **Résultat sur capture réelle** : vue de profil inchangée (côté filmé à 0.94-1.00), mais 60 des
  63 cycles de la vue de face reposaient sur des genoux inférés (0.32) — la déviation du genou
  droit passait de +2.6° à -0.3°, soit un changement de signe.
- **Conséquence matérielle** : les déviations de genoux vue de face ne sont pas exploitables avec
  le placement actuel de l'ESP32-CAM. Cadrage/éclairage à revoir avant d'utiliser cette métrique.

### Découpage des vidéos ✅
- `--start`/`--end` et `--front-start`/`--front-end`, appliqués **après** le cache de landmarks :
  ajuster les bornes se rejoue instantanément.
- Nécessaire en pratique : la montée et la descente du vélo faussaient la recommandation de
  prolongateurs (-111 mm sur la vidéo entière contre -48 mm sur le segment pédalé).

### Robustesse des cycles ✅
- `detect_cycles` autorise jusqu'à 152 rpm (`min_dist = fps * 0.4`), ce qui laisse passer de faux
  PMB. Sur une capture réelle, 1 cycle parasite de 0.39 s (médiane 0.79 s) faisait passer
  l'écart-type de `genou_extension_pmb` de 2.2° à 9.8°.
- `_filter_implausible_cycles` (`kinematics/cycles.py`) rejette tout cycle dont la durée s'écarte de
  plus de 35% (`CYCLE_DURATION_TOLERANCE`) de la médiane des cycles détectés, à partir de 3 cycles
  (`MIN_CYCLES_FOR_PLAUSIBILITY_FILTER` — sous ce seuil la médiane n'est pas assez fiable). Testé sur
  signaux synthétiques (`tests/test_cycles.py`) : faux PMB reproduisant le cas réel, variation de
  cadence naturelle (20%) préservée, pas de filtrage en dessous de 3 cycles.

### Phase 4 — UI web *(optionnel)* — MVP fait
- ✅ Streamlit (`src/bikefit/ui/`) : upload vidéo profil + face optionnelle, formulaire de config
  (côté, découpage, marqueur ArUco, vélo/coureur/profil cible), rapport HTML + recommandations affichés
  dans le navigateur. Réutilise `cli.analyze` directement (appel Python direct, pas de subprocess).
- Restent hors scope MVP, à ajouter si l'usage le justifie : squelette superposé (temps réel ou vidéo
  de debug intégrée), historique/comparaison de sessions dans l'UI, `measure-body`/`measure-setup`.

---

## 6. Optimisations implémentées

- **Cache Parquet** : la pose n'est calculée qu'une fois par vidéo (clé mtime+size).
- **Angles vectorisés** : `compute_angles` traite toutes les frames en une passe NumPy.
- **NaN propre** : les frames non détectées — et les landmarks inférés (visibilité < 0.5) —
  propagent NaN à travers filtre → angles → agrégation.
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
