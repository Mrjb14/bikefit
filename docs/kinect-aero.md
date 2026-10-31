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

## Installation (Windows) — à valider sur ta machine

Rien de ceci n'a encore été testé sur le matériel ; les étapes sont celles habituellement
documentées pour libfreenect, à confirmer et corriger ici une fois faites.

1. Brancher la Kinect (adaptateur secteur + USB 2.0 de préférence).
2. **Zadig** : installer le pilote libusb (WinUSB/libusbK) sur les périphériques
   « Xbox NUI Motor », « Xbox NUI Camera » (et « Xbox NUI Audio » s'il apparaît).
3. Compiler **libfreenect** (OpenKinect, CMake) avec la liaison Python
   (`wrappers/python`, Cython) pour la version de Python du projet, et rendre le
   module `freenect` importable depuis `uv run python`.
4. Test rapide : `uv run python -c "import freenect; print(freenect.sync_get_depth())"`.

Alternative si la compilation coince : Kinect SDK 1.8 de Microsoft (accepte la Kinect
Xbox 360 pour du développement) avec un petit exécutable C++ qui écrit le même format
(`depth/000000.png` 16 bits en mm alignée sur la couleur + `meta.json`) — le pipeline
ne dépend que du format, pas du pilote. Voir `bikefit.aero.recording`.

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
