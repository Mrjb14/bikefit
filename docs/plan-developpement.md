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
                                                               │
                        ┌──────────────┐   ┌──────────────┐    │
                        │  Interface   │◀──│  Rapport /   │◀───┘
                        │  (UI)        │   │  Export      │
                        └──────────────┘   └──────────────┘
```

### 2.1 Module Ingestion vidéo
- Entrée : fichiers vidéo (mp4/mov, 60 fps recommandé) ou flux webcam en direct.
- Responsabilités : décodage (OpenCV / PyAV), normalisation (rotation, resize), extraction des métadonnées (fps, résolution), synchronisation multi-angles si deux vidéos.
- Sortie : itérateur de frames horodatées.

### 2.2 Module Pose Estimation
- Moteur : **MediaPipe Pose** (Tasks API, modèle `pose_landmarker_heavy` pour la précision).
- Abstraction : interface `PoseEstimator` pour pouvoir remplacer par YOLO-pose ou autre sans toucher au reste (pattern Strategy).
- Sortie : par frame, 33 landmarks (x, y, z, visibilité) en coordonnées normalisées.
- Optimisation : traitement par batch, option de sous-échantillonnage spatial (la pose n'a pas besoin du 4K), mise en cache des résultats sur disque (un fichier `.parquet` de landmarks par vidéo) pour ne jamais recalculer.

### 2.3 Module Analyse cinématique (le cœur métier)
- **Filtrage** : lissage des landmarks (filtre Savitzky-Golay ou Butterworth passe-bas ~6 Hz) pour éliminer le bruit de détection.
- **Détection des cycles de pédalage** : suivi de la trajectoire de la cheville/pied → détection des points morts haut (PMH) et bas (PMB) par analyse des extrema de la coordonnée verticale.
- **Calcul d'angles** par frame : genou (hanche-genou-cheville), hanche (épaule-hanche-genou), tronc/horizontale, coude, cheville.
- **Agrégation** : pour chaque angle, distribution sur N cycles (min, max, moyenne, écart-type). Rejet des cycles aberrants.
- **Calibration métrique** : conversion pixels → millimètres via une référence connue (diamètre de roue 700c ≈ 622 mm de jante, ou marqueur ArUco imprimé collé sur le cadre — solution la plus robuste).
- Sortie : objet `FitMetrics` structuré (JSON-sérialisable).

### 2.4 Module Configuration vélo & matériel
- **Fiche vélo** (`BikeProfile`, Pydantic, stockée en YAML + SQLite) :
  - Identité : marque, modèle, référence, année, taille de cadre, numéro de série (optionnel).
  - Géométrie constructeur : stack, reach, angle de tube de selle, longueur de douille.
  - Transmission : longueur de manivelles (**indispensable** pour la conversion angle→mm et le calcul PMH/PMB), plateaux.
  - Périphériques : modèle de selle (longueur des rails), tige de selle (déport), cintre/prolongateurs (marque, réf, plages de réglage), potence (longueur, angle), cales/pédales.
  - Roues : diamètre réel (sert de référence de calibration de secours).
- **Réglages courants mesurés** (`BikeSetup`, versionné à chaque session) : hauteur de selle (axe pédalier → creux de selle), recul de selle, longueur/hauteur des pads, inclinaison des extensions, hauteur cintre-selle (drop).
- **Plages de réglage admissibles** par composant (min/max en mm) : permet au moteur de recommandation de détecter les butées ("recul demandé : 20 mm, disponible : 8 mm → proposer tige de selle déportée") et de vérifier la conformité UCI si souhaité (option).
- Un fichier YAML par vélo dans `config/bikes/`, réutilisable entre sessions ; chaque session d'analyse référence un vélo + un setup daté.

### 2.5 Moteur de recommandation
- Base de règles explicites (pas de ML au départ — les plages cibles CLM sont documentées) :
  - Extension genou PMB hors [140°, 150°] → ajustement hauteur de selle (règle ~3° ≈ 10 mm, affinée par la géométrie mesurée jambe/manivelle).
  - Angle hanche fermé < seuil → recul de selle ou remontée du poste.
  - Tronc, coudes, alignement frontal des genoux → recommandations dédiées.
- Chaque recommandation = {paramètre, direction, amplitude en mm, justification, confiance}.
- Le moteur consomme le `BikeProfile` (2.4) : longueur de manivelles pour affiner la conversion angle→mm, plages de réglage pour détecter les butées et proposer un changement de composant le cas échéant.
- Les plages cibles sont dans un fichier de config (YAML) modifiable, avec profils (confort / agressif).

### 2.6 Rapport & Export
- Rapport HTML/PDF : angles mesurés vs cibles, graphiques (courbes d'angle sur un cycle), frames annotées aux PMH/PMB, liste des réglages.
- En-tête du rapport : fiche du vélo (marque, modèle, taille) + setup mesuré au moment de la session.
- Historique des sessions pour comparer avant/après réglage (itération essentielle du fitting).

### 2.7 Interface utilisateur
- **v1 : CLI** (`bikefit analyze video.mp4 --view side`) — rapide à développer, suffisant pour valider le cœur.
- **v2 : UI web locale** (FastAPI + frontend léger, ou Streamlit pour aller vite) : upload vidéo, visualisation squelette superposé, lecture frame par frame, rapport interactif.

---

## 3. Stack technique

| Couche | Choix | Justification |
|---|---|---|
| Langage | Python 3.11+ | Écosystème CV, rapidité de dev |
| Vision | MediaPipe Tasks, OpenCV | Gratuit, précis, temps réel |
| Calcul | NumPy, SciPy, pandas | Filtrage, géométrie, stats |
| Calibration | OpenCV ArUco | Robuste, un marqueur imprimé suffit |
| Config | Pydantic + YAML | Validation forte des seuils |
| Stockage | Parquet (landmarks) + SQLite (sessions) | Simple, local, sans serveur |
| Rapport | Jinja2 → HTML, Plotly pour les graphes | Export facile en PDF |
| UI v2 | Streamlit (rapide) ou FastAPI+Svelte (propre) | Selon ambition |
| Tests | pytest + vidéos de référence | Non-régression sur les angles |
| Packaging | uv / pyproject.toml | Standard moderne |

---

## 4. Structure du projet

```
bikefit/
├── pyproject.toml
├── config/
│   ├── targets_tt.yaml        # plages cibles position CLM
│   ├── targets_road.yaml
│   └── bikes/
│       └── mon_clm.yaml       # fiche vélo : marque, réf, taille, géométrie, manivelles…
├── src/bikefit/
│   ├── ingest/                # lecture vidéo, sync multi-angles
│   ├── pose/                  # PoseEstimator + implémentations
│   ├── kinematics/            # filtrage, cycles, angles, calibration
│   ├── equipment/             # modèles BikeProfile / BikeSetup, plages de réglage
│   ├── recommend/             # moteur de règles
│   ├── report/                # génération rapport
│   ├── storage/               # cache landmarks, sessions SQLite
│   └── cli.py
├── tests/
│   ├── data/                  # courtes vidéos de référence annotées
│   └── ...
└── ui/                        # v2
```

---

## 5. Plan de développement par phases

### Phase 0 — Setup & spike technique (2-3 jours)
- Init repo, environnement, CI basique (lint + tests).
- Spike : MediaPipe sur une de tes vraies vidéos → vérifier que la détection est exploitable dans tes conditions (lumière, tenue, arrière-plan). **C'est le risque n°1 du projet, on le lève en premier.**
- Livrable : script jetable qui superpose le squelette sur la vidéo.

### Phase 1 — Pipeline profil, cœur métier (1-2 semaines)
- Ingestion + pose + cache des landmarks.
- Filtrage, détection des cycles, calcul des 5 angles clés, agrégation multi-cycles.
- CLI qui sort les métriques en JSON + vidéo annotée.
- Tests unitaires géométrie (angles sur squelettes synthétiques connus).
- **Jalon : les angles mesurés sont stables (±2°) entre deux vidéos de la même position.**

### Phase 2 — Recommandations, calibration & fiche vélo (1 semaine)
- Marqueur ArUco → échelle mm/pixel, conversion des écarts d'angle en réglages mm.
- Modèle `BikeProfile` + saisie de la fiche de ton CLM (`config/bikes/mon_clm.yaml`) : marque, référence, taille, longueur de manivelles, plages de réglage — consommée par le moteur de règles.
- Moteur de règles + config YAML des cibles CLM.
- Rapport HTML avec graphes, frames annotées et fiche vélo en en-tête.
- **Jalon : premier fitting réel complet sur ton vélo, vérifié au mètre ruban.**

### Phase 3 — Multi-angles & sessions (1 semaine)
- Vue de face : alignement des genoux, largeur coudes.
- `BikeSetup` versionné : chaque session enregistre les réglages mesurés du moment → traçabilité complète de l'évolution de la position.
- Historique SQLite, comparaison avant/après.
- **Jalon : boucle complète régler → refilmer → comparer.**

### Phase 4 — UI web (optionnelle, 1-2 semaines)
- Streamlit : upload, lecture annotée, rapport interactif, historique.

---

## 6. Optimisations prévues

- **Cache agressif** : les landmarks d'une vidéo ne sont calculés qu'une fois ; toute l'analyse en aval se rejoue en <1 s depuis le Parquet — itération rapide sur les règles sans re-traiter la vidéo.
- **Traitement à résolution réduite** pour la pose (720p suffit), frames originales conservées pour l'annotation.
- **Parallélisation** : décodage vidéo et inférence en pipeline (producteur/consommateur) ; multi-process si plusieurs vidéos.
- **Moyenne multi-cycles** plutôt que frame unique : précision et robustesse sans coût matériel.

---

## 7. Risques & parades

| Risque | Impact | Parade |
|---|---|---|
| Détection imprécise cheville/pied (chaussure) | Angles genou/cheville faux | Marqueurs contrastés, éclairage latéral, spike Phase 0 |
| Caméra non perpendiculaire au plan du vélo | Angles projetés biaisés | Check de perpendicularité via ArUco (pose 3D du marqueur), guide de placement caméra |
| Occlusion jambe opposée (vue profil) | Landmarks confondus | Filtrer par visibilité MediaPipe, ne garder que la jambe côté caméra |
| Conversion angle→mm approximative | Sur/sous-correction | Recommander des ajustements par petits pas + re-mesure (boucle Phase 3) |

---

## 8. Critères de succès v1

1. Écart-type < 2° sur chaque angle entre trois captures de la même position.
2. Un fitting complet (capture → rapport) en moins de 10 minutes.
3. Les réglages recommandés, appliqués et re-mesurés, ramènent les angles dans les plages cibles.
