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
| Ingest | `ingest/` | Lit un fichier, un flux MJPEG réseau ou une webcam locale, produit des frames horodatées |
| Pose | `pose/` | MediaPipe `pose_landmarker_heavy` (Tasks API), cache Parquet |
| Calibration | `calibration/` | Détecte un marqueur ArUco → `mm_per_pixel` (optionnel) |
| Kinematics | `kinematics/` | Filtre SG, détection cycles, 5 angles vectorisés, agrégation |
| Recommend | `recommend/` | Règles explicites : angles hors cible → delta mm par réglage |
| Report | `report/` | Rapport HTML interactif (Plotly), 5 graphes d'angles, schémas SVG profil + face |
| Storage | `storage/` | Sessions SQLite : historique, comparaison avant/après |
| Measure | `measure/` | Bootstrap des YAML personnels : mensurations (photo) et setup vélo (vidéo + ArUco) |
| Debug | `debug/` | Vidéo MP4 annotée (squelette + angles) pour debugger la détection, toute source confondue |

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

VIDEO : fichier (mp4/mov), flux MJPEG (http://... ou rtsp://...) ou webcam locale (webcam:0)

Options:
  --side TEXT          right | left — côté filmé (défaut: right)
  --duration FLOAT     Durée de capture (s) si VIDEO est un flux réseau ou une webcam (défaut: 15)
  --start FLOAT        Ignore les N premières secondes (fichier) — montée sur le vélo
  --end FLOAT          Ignore au-delà de N secondes (fichier) — descente du vélo
  --front-start FLOAT  Idem --start, pour la vidéo de face
  --front-end FLOAT    Idem --end, pour la vidéo de face
  --bike TEXT          Fiche vélo dans config/bikes/ (défaut: giant_trinity_2019)
  --profile TEXT       triathlon_XL | clm_long | clm_court (défaut: triathlon_XL)
  --report PATH        Rapport HTML de sortie
  --output PATH        Métriques JSON de sortie
  --marker-size FLOAT  Taille du marqueur ArUco en mm (ex. 100) — active la calibration
                       (fichier vidéo uniquement, pas de flux/webcam)
  --save-session       Sauvegarde la session dans ~/.bikefit/sessions.db
  --db PATH            Chemin alternatif pour la base SQLite
  --no-cache           Force le recalcul des landmarks MediaPipe (fichier uniquement)
  --front TEXT         Vidéo, flux MJPEG ou webcam (webcam:0) de la vue de face (Phase 3C)
  --front-duration FLOAT  Durée de capture (s) si --front est un flux réseau ou une webcam (défaut: 15)
  --debug-video PATH   Vidéo de debug (squelette + angles superposés), vue de profil
  --front-debug-video PATH  Vidéo de debug (squelette + métriques superposés), vue de face
```

`VIDEO` et `--front` ne peuvent pas pointer vers la même source (même fichier, même flux, même
index webcam) — la commande refuse de lancer les deux vues sur la même caméra.

Sans `--report`, le rapport est écrit dans `data/outputs/reports/<video>_<horodatage>.html` :
une analyse n'écrit jamais à côté de la vidéo source, et deux analyses de la même vidéo ne
s'écrasent pas.

### Découper la vidéo (`--start` / `--end`)

Une vidéo filmée seul contient toujours la montée et la descente du vélo. Ces phases hors
pédalage faussent lourdement les moyennes — mesuré sur une capture de 79 s :

| | Vidéo entière | Segment 20–70 s |
|---|---|---|
| Coude | 123,8° ± 25,1 | 108,1° ± 0,9 |
| Tronc | 24,9° ± 14,8 | 16,3° ± 0,5 |
| Reco prolongateurs | −111 mm | **−48 mm** |

```bash
python -m bikefit data/sources/video.mp4 --side right --start 20 --end 70
```

Le découpage s'applique **après** le cache de landmarks : ajuster les bornes se rejoue
instantanément, la vidéo n'est jamais réanalysée.

### `history` — liste les sessions

```
python -m bikefit history [--limit N] [--db PATH]
```

### `compare` — compare deux sessions avant/après réglage

```
python -m bikefit compare ID1 ID2 [--db PATH]
```

Affiche un tableau des deltas d'angles et les recommandations restantes.

### `measure-body` — bootstrap des mensurations depuis une photo

```
python -m bikefit measure-body PHOTO [OPTIONS]

Options:
  --rider TEXT   Fiche coureur dans config/rider/ (défaut: mensurations)
  --write        Met à jour les champs null du YAML (n'écrase jamais un champ déjà renseigné)
```

Estime `entrejambe`, `longueur_femur`, `longueur_tibia`, `longueur_buste`, `largeur_epaules`,
`largeur_hanches` depuis une photo debout (MediaPipe, world landmarks 3D), mis à l'échelle via
`taille` si renseignée. Précision indicative ±5-10 % — à valider au mètre ruban
(voir [`docs/procedure-mesures.md`](docs/procedure-mesures.md)).

### `measure-setup` — bootstrap du setup vélo depuis une vidéo

```
python -m bikefit measure-setup VIDEO --marker-size FLOAT [OPTIONS]

Options:
  --side TEXT          right | left — côté filmé (défaut: right)
  --bike TEXT           Fiche vélo dans config/bikes/ (défaut: giant_trinity_2019)
  --write               Met à jour les champs null du YAML
  --no-cache            Force le recalcul des landmarks MediaPipe
```

Calibre via ArUco puis estime le centre du boîtier de pédalier (fit de cercle sur la trajectoire
de la cheville) pour en déduire `hauteur_selle_bb`, `hauteur_guidon_bb`, `reach_prolongateurs`.
Précision indicative ±20 mm — validation au mètre ruban obligatoire avant usage.

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

## Sources vidéo : fichier, flux réseau ou webcam

`VIDEO` (vue de profil) et `--front` (vue de face, Phase 3C) acceptent chacun indifféremment :

- un **fichier** (mp4/mov) ;
- un **flux MJPEG réseau** (`http://...`, `https://...`, `rtsp://...` — ex. ESP32-CAM, firmware et
  procédure de flash dans [`firmware/`](firmware/README.md)) ;
- une **webcam locale** (`webcam:<index>`, ex. `webcam:0` pour la première webcam détectée par OpenCV).

Flux réseau et webcam n'ont pas de fin naturelle : la capture est bornée par `--duration`/
`--front-duration` (15 s par défaut). Un fichier vidéo se termine de lui-même, cache Parquet inclus ;
un flux/webcam n'est jamais mis en cache (rejoué à chaque lancement).

```bash
# Vue de profil en fichier, vue de face en webcam
python -m bikefit video.mp4 --side right --front webcam:0 --front-duration 30

# Les deux vues en webcam (deux périphériques distincts obligatoires)
python -m bikefit webcam:0 --side right --duration 60 --front webcam:1 --front-duration 60

# Vue de face depuis un flux MJPEG (ESP32-CAM), capture bornée à 30 s
python -m bikefit video.mp4 --side right --front http://192.168.1.42/stream --front-duration 30
```

`VIDEO` et `--front` ne peuvent pas être la même source — `webcam:0` des deux côtés (ou le même
fichier, ou la même URL) est rejeté au lancement, avant toute ouverture de caméra :

```
Erreur : --front pointe vers la même source que la vidéo de profil (fichier, flux ou webcam) —
impossible de lancer les deux vues sur la même caméra.
```

Pour la vue de face, les métriques apparaissent dans `FitMetrics` (`genou_deviation_gauche/droit`,
`largeur_coudes`) et un schéma vélo vue de face s'ajoute au rapport HTML. ⚠️ Le firmware ESP32-CAM ne
gère qu'un client à la fois : fermer tout onglet/lecteur déjà ouvert sur le flux avant de lancer la
capture. La calibration ArUco (`--marker-size`) n'est disponible que sur fichier vidéo.

## Vidéo de debug

`--debug-video`/`--front-debug-video` génèrent une vidéo MP4 avec le squelette détecté et les angles
superposés (HUD en incrustation), pour vérifier visuellement ce que le pipeline détecte réellement —
utile quand les métriques semblent aberrantes ou que le taux de détection est faible. Fonctionne avec
n'importe quelle source (fichier, flux réseau, webcam).

```bash
# Fichier — rejoue la vidéo (décodage seul, sans MediaPipe) avec les angles lissés finaux
python -m bikefit video.mp4 --side right --debug-video debug.mp4

# Webcam en direct — écrit pendant la capture, angles bruts par frame (pas de lissage possible)
python -m bikefit webcam:0 --side right --duration 30 --debug-video debug_live.mp4

# Vue de face
python -m bikefit video.mp4 --side right --front webcam:0 --front-debug-video debug_face.mp4
```

Deux comportements selon la source, documentés pour éviter toute confusion :

- **Fichier** : la vidéo de debug superpose les angles **lissés** (post-filtre Savitzky-Golay) — les
  valeurs exactes utilisées pour les recommandations. Rejeu léger (décodage seul, sans MediaPipe), donc
  rapide même quand les landmarks viennent du cache.
- **Flux réseau / webcam** : un flux live ne se rejoue pas. Le squelette et les angles affichés sont
  calculés **frame par frame, bruts** (sans lissage, qui nécessite l'ensemble de la capture) — normal
  de les voir plus "nerveux" que sur un rapport final.

## Structure du projet

```
bikefit/
├── config/
│   ├── targets_tt.yaml          # angles cibles (clm_court/long, triathlon_XL) + coefficients
│   ├── bikes/                   # fiches vélo personnelles (gitignorées)
│   └── rider/                   # mensurations personnelles (gitignorées)
├── docs/
│   ├── plan-developpement.md
│   ├── procedure-mesures.md     # procédure de prise de mesures + validation ruban
│   └── aruco_marker_100mm.png   # marqueur à imprimer (100 mm, DICT_4X4_50 ID=0)
├── firmware/
│   ├── README.md                # installation/flash ESP32-CAM
│   └── esp32cam_mjpeg/          # sketch Arduino, serveur MJPEG HTTP (vue de face)
├── data/                        # données locales, entièrement gitignoré
│   ├── sources/                 # vidéos et photos d'entrée (seuls fichiers non regénérables)
│   └── outputs/                 # reports/ · metrics/ · debug/ · annotated/ — tout le généré
├── scripts/
│   └── spike_bikefit.py         # script autonome Phase 0
└── src/bikefit/
    ├── ingest/                  # VideoReader (fichier, flux MJPEG réseau ou webcam), TimedFrame
    ├── pose/                    # PoseEstimator (Protocol) + MediaPipe + LandmarkCache
    ├── kinematics/              # filter, cycles, angles, aggregate, frontal, visibility, FitMetrics
    ├── calibration/             # ArUco — detect_marker_size_px, calibrate_from_video
    ├── equipment/               # BikeProfile, BikeSetup, loader YAML
    ├── recommend/               # moteur de règles → FitReport + Recommendation
    ├── report/                  # rapport HTML Jinja2 + Plotly (dark theme, schémas profil + face)
    ├── storage/                 # Session + SessionStore (SQLite)
    ├── measure/                 # Bootstrap YAML : mensurations (photo) + setup vélo (vidéo/ArUco) + annotation visuelle
    ├── debug/                   # Vidéo de debug annotée (squelette + HUD d'angles), toute source
    └── cli.py                   # Typer — analyze (--front, --debug-video) / history / compare / measure-*
```

Les données runtime vivent **hors du dépôt**, dans `~/.bikefit/` : `models/` (modèle MediaPipe
téléchargé au premier lancement), `cache/landmarks/` (Parquet par vidéo) et `sessions.db`.

## Tests

```bash
python -m pytest        # 198 tests — angles, filtre SG, cycles, visibilité/fiabilité, ArUco,
                         # recommandations, storage, plan frontal, ingest MJPEG/webcam, measure/,
                         # report HTML, CLI, vidéo de debug (données synthétiques)
uv run mypy src
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

CI : matrice `ubuntu` (Python 3.11, 3.12) **et `windows`** — plateforme de dev principale, seule à
révéler les bugs d'encodage console et de séparateurs de chemin. Couverture minimale : 65 %.

## Documentation

- [Plan de développement & architecture](docs/plan-developpement.md)
- [Procédure de prise de mesures](docs/procedure-mesures.md) — `measure-body` / `measure-setup` + validation au mètre ruban
- [CLAUDE.md](CLAUDE.md) — contexte pour Claude Code

## Statut

- [x] Phase 0 — spike MediaPipe (Tasks API v1.0)
- [x] Phase 1 — pipeline ingest → pose → kinematics → CLI, jalon ±2° atteint
- [x] Phase 2 — moteur de règles, fiche vélo, rapport HTML (coefficients empiriques)
- [x] Phase 3A — calibration ArUco : `mm_per_pixel` détecté, stocké en session *(marqueur non encore imprimé)*
- [x] Phase 3B — sessions SQLite : `history`, `compare`, `--save-session`
- [x] `measure/` — bootstrap `measure-body`/`measure-setup` (mensurations + setup vélo, photo/vidéo annotée)
- [ ] Phase 3A+ — mesure automatique du vélo (BB → selle, reach, stack) via 2ᵉ marqueur ArUco
- [x] Phase 3C — vue de face (`--front`, fichier ou flux MJPEG ESP32-CAM) : déviation genoux, largeur coudes, firmware ESP32-CAM
- [x] Vue de profil en flux réseau/webcam (symétrique à `--front`) + garde anti-double-source
- [x] Vidéo de debug (`--debug-video`/`--front-debug-video`) : squelette + angles superposés, toute source
- [x] Découpage `--start`/`--end` — exclut la montée/descente du vélo, appliqué après le cache
- [x] Fiabilité des mesures — les landmarks *inférés* par MediaPipe sont exclus des angles, et chaque
      métrique porte sa visibilité + son taux de frames exploitées (voir ci-dessous)
- [ ] Filtre de plausibilité sur la durée des cycles — un faux PMB fait encore passer l'écart-type
      du genou de 2,2° à 9,8°
- [ ] Phase 4 — UI Streamlit (optionnel)

## Fiabilité des mesures

MediaPipe *infère* les landmarks masqués : sur une vue de profil, le côté opposé à la caméra
descend typiquement à 0,2–0,3 de `visibility`. Le squelette reste plausible, mais les coordonnées
sont devinées.

Un angle dont l'un des sommets passe sous le seuil (0,5) est donc **exclu** plutôt que calculé sur
des positions inventées, et chaque métrique porte sa `visibility` et son `frames_kept` — affichés
dans le rapport HTML et en avertissement CLI dès qu'une mesure devient douteuse.

Mesuré sur une capture réelle : vue de profil inchangée (côté filmé à 0,94–1,00), mais 60 des
63 cycles de la **vue de face** reposaient sur des genoux inférés (genou gauche à 0,32) — la
déviation du genou droit passait de +2,6° à −0,3°, soit un changement de signe.
