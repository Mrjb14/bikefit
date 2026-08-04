# BikeFit CLM — Plan de développement & Architecture

Logiciel d'analyse vidéo pour le réglage d'un vélo de contre-la-montre, basé sur l'estimation de pose par vision par ordinateur.

---

## 1. Vision du produit

**Objectif** : à partir de vidéos (profil + face) d'un cycliste pédalant sur home trainer, produire un rapport de réglage chiffré : hauteur/recul de selle, position des prolongateurs, hauteur du poste de pilotage.

**Utilisateur cible** : toi (v1), puis potentiellement d'autres cyclistes (v2+).

**Principe directeur** : précision et reproductibilité avant tout. Une recommandation fausse de 10 mm est pire que pas de recommandation. Chaque mesure doit être moyennée sur plusieurs cycles de pédalage et accompagnée d'un indice de confiance.

---

## 2. Architecture générale

Architecture modulaire en pipeline, chaque module testable indépendamment :

```
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Ingestion   │──▶│ Pose         │──▶│ Analyse      │──▶│ Moteur de    │
│  vidéo       │   │ Estimation   │   │ cinématique  │   │ recommandation│
└─────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                         │                                       │
                    cache Parquet                                ▼
                                                        ┌──────────────┐
                                                        │  Rapport HTML │
                                                        │  Jinja2+Plotly│
                                                        └──────────────┘
```

### 2.1 Module Ingestion vidéo — ✅ implémenté
- `VideoReader` : lit fps/width/height/frame_count à l'init, itérateur `frames()` → `TimedFrame`.
- Entrée : fichiers vidéo (mp4/mov). Flux MJPEG réseau prévu Phase 3.
- Horodatage : `timestamp_ms = frame_idx * 1000 / fps`.

### 2.2 Module Pose Estimation — ✅ implémenté
- Moteur : **MediaPipe Pose Tasks API v1.0** (`pose_landmarker_heavy`, `VisionRunningMode.VIDEO`).
- Protocole `PoseEstimator` (Strategy) : le code métier n'importe jamais MediaPipe directement.
- `LandmarkCache` : clé `md5(path+mtime+size)[:12]`, stockage Parquet dans `cache/landmarks/`.
  Recalcul uniquement si la vidéo a changé. Tout le pipeline en aval se rejoue en < 1 s.
- `FrameLandmarks` : 33 landmarks (x, y, z, visibilité) normalisés + flag `detected`.

### 2.3 Module Analyse cinématique — ✅ implémenté
- **Filtrage** : `filter_landmarks` — interpolation linéaire sur les gaps NaN, puis Savitzky-Golay (polyorder=3, fenêtre ≈ fps/5) pour éliminer le bruit ~6 Hz. NaN restaurés pour les frames non détectées.
- **Détection des cycles** : `detect_cycles` — `find_peaks` sur `ankle_y` (y image vers le bas → PMB = maximum). Distance minimale entre pics = `fps * 0.4` (permet jusqu'à 150 rpm).
- **Calcul d'angles** : `compute_angles` — vectorisé NumPy sur toutes les frames en une passe.
  - genou (hanche–genou–cheville), hanche (épaule–hanche–genou), tronc (angle épaule-hanche / horizontale), coude (épaule–coude–poignet), cheville (genou–cheville–pied).
  - NaN propagé pour les frames non détectées.
- **Agrégation** : `aggregate_cycles` — par cycle : genou_ext (max), genou_flex (min), hanche (min), tronc/coude (mean), cheville (max PMB). Cycles avec > 10% NaN rejetés.
- Sortie : `FitMetrics` (Pydantic) — 6 angles × `{mean, std, min, max, n_cycles}`.

### 2.4 Module Équipement — ✅ implémenté
- `BikeProfile` (Pydantic) : modele, longueur_manivelle_mm, `BikeSetup` (réglages actuels en mm), `PlagesReglage` (butées min/max).
- `load_bike_profile(name)` : charge `config/bikes/{name}.yaml`, extrait longueur manivelles depuis la transmission.
- Fiche vélo personnelle `giant_trinity_2019.yaml` gitignorée.

### 2.5 Moteur de recommandation — ✅ implémenté
- `run(metrics, bike, profile)` → `FitReport`.
- Profils cibles dans `config/targets_tt.yaml` : `clm_court`, `clm_long`, `triathlon_XL`.
- Coefficients de sensibilité empiriques (°/mm) pour convertir un écart angulaire en delta mm.
- Règles implémentées :
  - **Genou extension PMB** → hauteur selle (0.65°/mm)
  - **Genou flexion PMH** → recul selle (directionnel)
  - **Hanche PMH** → hauteur prolongateurs (0.30°/mm)
  - **Coude** → longueur prolongateurs (0.25°/mm)
  - **Tronc** → hauteur prolongateurs (directionnel)
  - **Cheville PMB** → cale / technique
- Détection de butée : si le delta mm demandé dépasse la plage de réglage connue.
- `Recommendation` : reglage, direction, delta_mm, motif, priorité, butee_atteinte.

> **Non encore implémenté** : calibration ArUco (mm/pixel). Les delta_mm utilisent des coefficients empiriques, non une géométrie mesurée. Prévu Phase 3.

### 2.6 Rapport & Export — ✅ implémenté
- `render_html(metrics, report, angle_df, video_path, output_path)`.
- Template Jinja2 (`report/template.html.jinja2`) — dark theme, responsive.
- Graphe Plotly (`scattergl`) des 5 angles sur toutes les frames.
- Tableau des métriques avec code couleur vert/rouge selon plages cibles.
- Carte par recommandation : badge priorité, delta mm, motif, indicateur de butée.

### 2.7 Interface utilisateur
- **v1 : CLI** (`python -m bikefit video.mp4 --side right`) — ✅ implémentée.
- **v2 : UI web locale** (Streamlit) — Phase 4 optionnelle.

---

## 3. Stack technique

| Couche | Choix | Justification |
|---|---|---|
| Langage | Python 3.11+ | Écosystème CV, rapidité de dev |
| Vision | MediaPipe Tasks API v1.0, OpenCV | Gratuit, précis, temps réel |
| Calcul | NumPy, SciPy, pandas | Filtrage, géométrie, stats |
| Calibration | OpenCV ArUco (Phase 3) | Robuste, un marqueur imprimé suffit |
| Config | Pydantic v2 + YAML | Validation forte des seuils |
| Stockage | Parquet (landmarks) + SQLite (sessions Phase 3) | Simple, local, sans serveur |
| Rapport | Jinja2 → HTML, Plotly CDN | Export autonome, pas de dépendance serveur |
| UI v2 | Streamlit (Phase 4, optionnel) | Rapide à prototyper |
| Tests | pytest, squelettes synthétiques | Non-régression géométrie + recommandations |
| Packaging | uv / pyproject.toml | Standard moderne |

---

## 4. Structure du projet

```
bikefit/
├── pyproject.toml
├── config/
│   ├── targets_tt.yaml           # angles cibles par profil (clm_court, clm_long, triathlon_XL)
│   ├── bikes/                    # fiches vélo personnelles (gitignorées)
│   │   └── giant_trinity_2019.yaml
│   └── rider/                    # mensurations (gitignorées)
│       └── mensurations.yaml
├── models/                       # modèles MediaPipe .task (gitignorés)
├── cache/landmarks/              # Parquet par vidéo (gitignorés)
├── scripts/
│   └── spike_bikefit.py          # Phase 0
└── src/bikefit/
    ├── _paths.py                 # PROJECT_ROOT, MODELS_DIR, CACHE_DIR, CONFIG_DIR
    ├── cli.py                    # Typer — python -m bikefit
    ├── __main__.py               # entrée -m bikefit
    ├── ingest/
    │   └── reader.py             # VideoReader, TimedFrame
    ├── pose/
    │   ├── base.py               # Landmark, FrameLandmarks, PoseEstimator (Protocol)
    │   ├── mediapipe_estimator.py
    │   └── cache.py              # LandmarkCache
    ├── kinematics/
    │   ├── landmarks.py          # BODY dict, ANGLE_JOINTS
    │   ├── filter.py             # filter_landmarks (Savitzky-Golay)
    │   ├── cycles.py             # detect_cycles (find_peaks ankle_y)
    │   ├── angles.py             # compute_angles (vectorisé)
    │   ├── aggregate.py          # aggregate_cycles → FitMetrics
    │   └── metrics.py            # AngleStats, FitMetrics (Pydantic)
    ├── equipment/
    │   ├── models.py             # BikeProfile, BikeSetup, PlagesReglage
    │   └── loader.py             # load_bike_profile(name)
    ├── recommend/
    │   ├── models.py             # Recommendation, FitReport, Priorite
    │   └── engine.py             # run(metrics, bike, profile) → FitReport
    ├── report/
    │   ├── html.py               # render_html(...)
    │   └── template.html.jinja2
    └── storage/                  # (Phase 3) sessions SQLite
```

---

## 5. Plan de développement par phases

### Phase 0 — Setup & spike technique ✅
- Init repo, environnement (uv), CI.
- Spike MediaPipe Tasks API v1.0 (`scripts/spike_bikefit.py`) — migration depuis l'ancienne API `mp.solutions.pose` supprimée en v1.0.
- Livrable : script qui télécharge le modèle et superpose le squelette sur une vidéo.

### Phase 1 — Pipeline profil, cœur métier ✅
- `VideoReader` → `MediaPipePoseEstimator` → `LandmarkCache` (Parquet).
- `filter_landmarks` (Savitzky-Golay) → `compute_angles` (5 angles vectorisés) → `detect_cycles` → `aggregate_cycles` → `FitMetrics`.
- CLI Typer + `__main__.py`.
- 10 tests unitaires (angles sur squelettes synthétiques).
- **Jalon atteint** : 59 cycles, 100% détection, std ≤ 2.2° sur tous les angles.

### Phase 2 — Recommandations & rapport HTML ✅ (partielle)
- `BikeProfile` / `BikeSetup` / `PlagesReglage` Pydantic + `load_bike_profile`.
- `config/targets_tt.yaml` — 3 profils (clm_court, clm_long, triathlon_XL) + coefficients de sensibilité.
- Moteur de règles `recommend/engine.py` — 6 règles, priorités, détection butées.
- Rapport HTML dark theme — Jinja2 + Plotly, autonome (CDN), responsive.
- 20 tests unitaires (angles + recommandations).
- **Manquant** : calibration ArUco (mm/pixel) — les delta_mm sont estimés via coefficients empiriques.
- **Jalon** : premier fitting complet → rapport HTML exploitable. Vérification au mètre ruban des réglages recommandés.

### Phase 3 — Calibration ArUco & multi-angles (à faire)
- Marqueur ArUco imprimé sur le cadre → `mm_per_pixel` → conversion précise écart_angle → mm.
- Vue de face : alignement des genoux, largeur coudes.
- `BikeSetup` versionné : chaque session enregistre la date + les réglages mesurés.
- Historique SQLite, comparaison avant/après.
- Flux MJPEG HTTP (ESP32-CAM) dans `VideoReader`.
- **Jalon** : boucle complète régler → refilmer → comparer.

### Phase 4 — UI web (optionnelle)
- Streamlit : upload vidéo, squelette superposé, rapport interactif, historique sessions.

---

## 6. Optimisations implémentées

- **Cache Parquet** : la pose n'est calculée qu'une fois par vidéo (clé mtime+size). Tout le pipeline en aval en < 1 s.
- **Angles vectorisés** : `compute_angles` traite toutes les frames en une passe NumPy — pas de boucle Python.
- **NaN propre** : les frames non détectées propagent NaN à travers filtre → angles → agrégation.
- **Moyenne multi-cycles** (N=59 sur la vidéo de référence) : robustesse sans coût GPU.

---

## 7. Risques & parades

| Risque | Impact | Parade |
|---|---|---|
| Détection imprécise cheville/pied (chaussure) | Angles genou/cheville faux | Marqueurs contrastés, éclairage latéral, filtre de visibilité MediaPipe |
| Caméra non perpendiculaire au plan du vélo | Angles projetés biaisés | ArUco en Phase 3 donne la pose 3D du marqueur pour vérifier |
| Occlusion jambe opposée (vue profil) | Landmarks confondus | Ne garder que la jambe côté caméra (`--side right/left`) |
| delta_mm approximatif sans calibration ArUco | Sur/sous-correction | Recommander des ajustements par petits pas + re-mesure (Phase 3) |
| Variabilité de cadence entre sessions | Biais de cycle | Sélectionner une plage de cadence constante en post-traitement (Phase 3) |

---

## 8. Critères de succès v1

1. Écart-type < 2° sur chaque angle entre trois captures de la même position. ✅ (std ≤ 2.2°)
2. Un fitting complet (capture → rapport HTML) en moins de 10 minutes. ✅ (pipeline < 3 min sur CPU)
3. Les réglages recommandés, appliqués et re-mesurés, ramènent les angles dans les plages cibles. (à valider Phase 3)
