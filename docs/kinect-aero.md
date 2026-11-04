# Aéro à la Kinect — surface frontale, CdA, comparaison de positions

Objectif : une « soufflerie du pauvre » sur home trainer. La Kinect Xbox 360 filme le
cycliste **de face** ; chaque pixel porte sa distance, ce qui donne la **surface
frontale A** en m², sans marqueur. Calibrée une fois sur route, elle donne un **CdA
estimé**, puis des watts et des secondes gagnées entre deux positions.

## Ce que ça mesure, ce que ça ne mesure pas

| | Mesuré | Comment |
|---|---|---|
| Surface frontale A (m²) | Oui | Profondeur Kinect, moyennée sur les cycles de pédalage |
| Écart relatif entre positions | Oui | Avec test de significativité (bruit de mesure) |
| Cd (forme : tête, épaules, casque, tissu) | **Non** | Invisible pour une caméra → calibré sur route |
| CdA, watts, temps | Estimé | A × Cd calibré — hypothèse : Cd constant entre positions |
| Vent de travers (lacet) | Non | Une soufflerie le teste, pas ce montage |

Hypothèse à garder en tête : rentrer la tête derrière les épaules peut faire gagner
en Cd sans réduire A. Ce gain-là échappe à la Kinect.

## Matériel

- Kinect Xbox 360 (v1) + **adaptateur secteur/USB** (vendu à part, indispensable :
  le port USB du PC n'alimente pas le capteur ; la version « Xbox 360 S » a un
  connecteur propriétaire, il faut l'adaptateur correspondant).
- Le marqueur ArUco du projet (`docs/aruco_marker_100mm.png`, ID=0) collé sur un carton
  plat, pour vérifier les intrinsèques.
- Pour la calibration Cd : un capteur de puissance sur le vélo, et un parcours route.

## Installation (Windows)

### Voie testée : Kinect SDK 1.8 (recommandée)

Validée sur matériel réel (Kinect Xbox 360 + adaptateur secteur, Windows 11) : 30 fps,
profondeur alignée sur la caméra couleur, ~78 % de pixels valides dans une pièce.

1. Installer le **Kinect SDK 1.8** de Microsoft (il accepte la Kinect Xbox 360 pour le
   développement) ; la variable `KINECTSDK10_DIR` doit exister.
2. Brancher la Kinect (adaptateur secteur + USB 2.0).
3. Construire l'outil de capture (compilateur C# de .NET Framework, pas de Visual Studio) :
   `powershell -File tools/kinect_sdk/build.ps1`
4. Capturer : `uv run python scripts/kinect_capture.py data/sources/aero/test --duration 20`
   (`--driver auto` choisit le SDK dès que `kinect_grab.exe` existe).

`tools/kinect_sdk/KinectGrab.cs` lit profondeur + couleur, aligne la profondeur sur la
couleur (`MapColorFrameToDepthFrame`) et pousse des paquets binaires que
`kinect_capture.py` écrit au format `bikefit.aero.recording`.

### Voie alternative : libfreenect (non testée)

Zadig (pilote libusb sur « Xbox NUI Motor/Camera »), compilation de libfreenect + liaison
Python `freenect`, puis `--driver freenect`. Remplace le pilote Microsoft : incompatible
avec la voie SDK sur la même machine.

## Installation physique

```
   [Kinect]  ── ~2 m ──>  [roue avant]  ...  [cycliste]  ...  [home trainer]
   hauteur ≈ buste, dans l'axe du vélo, de niveau
```

- **Distance ~2 m** de la roue avant : champ vertical de 43°, il faut voir du sol au casque.
  La Kinect ne mesure rien à moins de ~0,8 m ; au-delà de ~3 m l'erreur grandit vite.
- **Dans l'axe du vélo**, centrée, à hauteur de buste : un décalage gonfle la surface
  mesurée (on voit le flanc). Le cycliste doit être au **centre de l'image** (la
  profondeur de référence est lue au centre).
- **De niveau** : le retrait du sol suppose une caméra à peu près horizontale.
- **Rien d'autre** dans la bande de profondeur du cycliste (personne, ventilateur…) :
  la plus grande forme dans la bande est prise pour le cycliste.
- **Pas de soleil direct** (il noie le motif infrarouge). Intérieur : aucun souci.
- **Ne plus bouger la Kinect** entre deux positions comparées : même montage = même biais.

## Protocole de capture

Par position : ~60 s de pédalage stabilisé, même cadence (~85-90 rpm), même puissance,
regard comme en course. Faire **deux captures** de la même position au début : leur
écart doit être « non significatif » — c'est la mesure de la répétabilité.

```bash
# Capture (une position = un dossier)
python scripts/kinect_capture.py data/sources/aero/2026-11-02_ref --duration 60
python scripts/kinect_capture.py data/sources/aero/2026-11-02_pads_-10mm --duration 60

# Analyse + comparaison (la 1re position sert de référence)
python -m bikefit aero data/sources/aero/2026-11-02_ref data/sources/aero/2026-11-02_pads_-10mm \
  --debug-dir data/outputs/debug --output data/outputs/metrics/aero_2026-11-02.json

# Ignorer le début/fin (mise en route), forcer la bande de profondeur si besoin
python -m bikefit aero data/sources/aero/ref --start 5 --end 55 --near-mm 1600 --far-mm 3200
```

**Toujours regarder la vidéo de contrôle** (`--debug-dir`) avant de croire un chiffre :
vert = compté, rouge = trou de profondeur bouché (compté aussi), gris = hors silhouette.
Le sol, le mur ou un objet en vert = segmentation fausse.

## Vérifier les intrinsèques (fx) au marqueur ArUco

La surface varie en 1/fx² : 2 % d'erreur sur fx = 4 % sur la surface. Les valeurs
d'usine (`fx = 525`) sont une approximation. Contrôle :

1. Tenir le marqueur (carton plat) face à la Kinect à ~1,5 m, 10 s, et capturer.
2. `python -m bikefit aero data/sources/aero/marqueur --marker-size 100`

Résultat `OK` si fx mesuré à ±2 %. Sinon, corriger `intrinsics.fx`/`fy` dans les
`meta.json` (ou dans `KINECT_V1_REGISTERED`). Pour les **comparaisons** entre positions
du même montage, une erreur sur fx s'annule ; elle compte pour un CdA absolu.

## Calibrer le Cd sur route (méthode Chung)

1. Rouler la **position de référence** sur une boucle plate, sans vent, plusieurs tours,
   avec capteur de puissance (fichier .fit avec puissance + vitesse).
2. Calculer le CdA par élévation virtuelle (ex. Aerolab de Golden Cheetah).
3. Capturer **la même position** à la Kinect, puis :

```bash
python -m bikefit aero data/sources/aero/ref --cda-terrain 0.245 --write
```

`--write` remplit la section `aero` de `config/rider/mensurations.yaml` (champs `null`
seulement). Les analyses suivantes relisent ce Cd et affichent CdA, watts à
`--speed-kmh` (36 par défaut) et temps sur `--distance-km` (180 par défaut).

Le Cd calibré absorbe les biais constants du montage (vélo et home trainer dans le
champ, fx approximatif) : **il n'est valable que pour ce montage de capture**. Un Cd
hors 0,45-1,10 est signalé comme suspect.

## Critères de l'essai (avant d'en faire un outil de réglage)

| Critère | Seuil | Où le lire |
|---|---|---|
| Trous de profondeur sur la silhouette (tenue de home trainer) | ≤ 10 % | `trous` dans la sortie |
| Répétabilité : 2 captures de la même position | écart « non significatif », < 1 % | `vs [1]` |
| Sensibilité : pads -10 à -20 mm | écart significatif | `vs [1]` |
| fx au marqueur | ±2 % | `--marker-size` |
| Cd calibré | 0,45-1,10 | `--cda-terrain` |

## Limites connues

- Projection : A = Σ z²/(fx·fy) approxime la projection orthographique dans l'axe du
  déplacement ; exacte caméra loin et centrée, approximative de biais.
- Les « pixels volants » de bord (profondeur intermédiaire entre cycliste et fond)
  peuvent ajouter un liseré ; l'ouverture morphologique en retire l'essentiel.
- Un trou de profondeur **sur le bord** de la silhouette (bras noir sur fond lointain)
  n'est pas récupérable : il n'est pas entouré par le cycliste. Il apparaît comme une
  encoche dans la vidéo de contrôle.
- La cadence est détectée sur la surface elle-même (autocorrélation) ; sans période
  nette, moyenne sur des fenêtres d'1 s (signalé).


## Séance d'entraînement : `bikefit seance`

Une commande remplace capture + découpage + analyse :

```bash
python -m bikefit seance --set-reference                # 1re fois : position de référence
python -m bikefit seance                                # séances suivantes : A vs référence
python -m bikefit seance --every-min 20 --times 5       # longue sortie : dérive de la position
python -m bikefit seance --profile-source webcam:0      # + angles de profil (caméra de profil)
```

Déroulé : la Kinect se remet à l'**horizontale** (moteur), compte à rebours pendant lequel
tu te mets en position et pédales, **centrage automatique** (le moteur incline la Kinect pour
centrer le cycliste, ±15° max), vérification du cadrage (distance 1,2-3,2 m, cycliste centré,
silhouette entière, sol visible), capture, **détection automatique du pédalage** (pas de
`--start/--end`), surface frontale, comparaison à la référence, résumé en quelques lignes.
Une fenêtre en direct (couleur | profondeur, avec « REC t / durée ») s'ouvre pendant le cadrage
et chaque capture, pour vérifier à l'œil que tout tourne (`--no-preview` pour la couper). Un cadrage
refusé s'explique ; `--force` capture quand même, `--no-auto-tilt` coupe le moteur.

Longue sortie : les cartes de profondeur pèsent ~0,5 Go/min, donc pas de capture continue.
`--every-min` fait une fenêtre de 60 s toutes les N minutes et affiche à la fin la dérive de la
surface (une hausse significative = tu te redresses ou écartes les coudes avec la fatigue).

Inclinaison : la surface projetée dépend légèrement de l'angle de la Kinect (cos θ). L'angle est
enregistré avec la référence ; `seance` avertit s'il diffère de plus de 3°.
Sans Cd calibré (cf. plus haut) : surfaces et écarts relatifs, pas de watts.
