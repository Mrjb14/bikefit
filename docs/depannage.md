# Dépannage

Guide par symptôme pour les problèmes rencontrés en pratique avec BikeFit. Pour la
procédure normale de prise de mesures, voir [`procedure-mesures.md`](procedure-mesures.md) ;
pour le flash de l'ESP32-CAM, voir [`firmware/README.md`](../firmware/README.md).

---

## Environnement

### `python -m bikefit` échoue avec `ModuleNotFoundError` (typer, mediapipe, ...)

Le `python` du PATH système ne pointe pas forcément vers le venv du projet — surtout
s'il existe un autre worktree Python (`git worktree`) ou un install global. Vérifier :

```bash
uv run python -c "import bikefit; print(bikefit.__file__)"
```

Si le chemin affiché ne pointe pas vers `.../bikefit/src/bikefit/__init__.py` de **ce**
dossier, c'est le symptôme. **Toujours préfixer par `uv run`** dans ce projet :

```bash
uv run python -m bikefit ...
uv run bikefit ...
uv run pytest
```

### Premier lancement lent / téléchargement du modèle MediaPipe

Normal : `pose_landmarker_heavy.task` (~30 Mo) est téléchargé automatiquement dans
`~/.bikefit/models/` au premier lancement qui en a besoin. Nécessite une connexion
réseau une seule fois ; les lancements suivants sont locaux.

---

## Calibration ArUco (marqueur ID=0, `--marker-size`)

**Symptôme** : `Avertissement : aucun marqueur ArUco détecté.` ou calibration marquée
« peu fiable » (`is_reliable=False`, `detection_rate < 50%` ou moins de 5 frames).

Causes les plus fréquentes, par ordre de probabilité :

1. **Mauvaise taille d'impression** — vérifier au double-décimètre que le marqueur
   imprimé fait exactement 100 mm de côté (une impression « ajustée à la page » fausse
   silencieusement toute la conversion mm/px).
2. **Marqueur pas plan** — un pli ou une courbure sur le tube de selle déforme les
   coins détectés et rend la mesure de taille bruitée.
3. **Hors du champ ou masqué** trop souvent par la jambe/le bras pendant le pédalage.
4. **Éclairage** — contre-jour ou reflet sur le marqueur (plastifier mat, pas brillant).
5. **ID confondu** — si un autre marqueur (ex. le BB, ID=1) est aussi dans le champ,
   `calibrate_from_video` ne regarde que l'ID=0 par construction ; vérifier que le bon
   marqueur est bien étiqueté ID=0 à l'impression (`docs/aruco_marker_100mm.png`).

## Marqueur BB (ID=1, `measure-setup`)

**Symptôme** : `Marqueur BB (ID=1) non détecté ou peu fiable -> BB estimé par fit de
cercle (±20 mm)` — repli automatique, pas une erreur bloquante, mais moins précis.

Mêmes causes que ci-dessus (impression, planéité, éclairage), plus spécifiquement :
occlusion trop fréquente par la manivelle/le pied à chaque tour (normal en passage,
problématique si constant — repositionner le marqueur hors de la trajectoire de la
manivelle). Si le repli ±20 mm ne suffit pas, pointer le pixel à la main :

```bash
uv run bikefit measure-setup video.mp4 --marker-size 100 --bb-px 812,540
```

(coordonnées en résolution vidéo originale — zoomer sur `data/outputs/annotated/<video>_annotated.jpg`
pour les lire ; caméra fixe obligatoire, cf. `procedure-mesures.md`).

---

## Détection de pose / métriques peu fiables

**Symptôme** : `Avertissement : métriques peu fiables (points inférés par le
modèle)` — un angle repose majoritairement sur des landmarks devinés par MediaPipe
(`visibility < 0.5`), typiquement le côté **opposé** à la caméra.

- En vue de profil : normal si `--side` correspond bien au côté filmé (le côté
  caché doit être exclu par construction) — vérifier que `--side` n'est pas inversé.
- En vue de face : cas connu et documenté (CLAUDE.md) sur le matériel actuel —
  repositionner/rééclairer l'ESP32-CAM avant de faire confiance à la déviation genoux.
- Cause générale : contre-jour, vêtements de la même couleur que le fond, cadrage
  trop serré (articulation hors champ).

## Cycles de pédalage

**Symptôme** : `Avertissement : moins de 3 cycles — résultats peu fiables.`

La vidéo est trop courte, ou le pédalage n'était pas continu sur la portion analysée.
Filmer au moins 60 s à cadence stable (voir `procedure-mesures.md`).

**Symptôme** : écart-type élevé sur un angle malgré ≥3 cycles détectés — un faux PMB
(détection parasite) peut s'être glissé. `_filter_implausible_cycles` rejette
automatiquement tout cycle s'écartant de plus de 35 % de la durée médiane, mais
seulement **à partir de 3 cycles détectés** (en dessous, la médiane n'est pas fiable).
Avec 2 cycles ou moins, le filtre ne peut pas s'appliquer — refilmer plus long.

---

## Cache de landmarks (`~/.bikefit/cache/landmarks/`)

**Symptôme** : une vidéo déjà analysée relance pourtant tout le calcul MediaPipe au
lieu de rejouer le cache.

La clé de cache est `md5(chemin_résolu + mtime + size)[:12]` (`pose/cache.py`) — **tout
changement** de ces trois éléments invalide le cache, y compris déplacer/renommer le
fichier source sans changer son contenu. Pour éviter une ré-inférence complète après un
déplacement, calculer la nouvelle clé et renommer le `.parquet` en conséquence :

```bash
uv run python -c "
from pathlib import Path
from bikefit.pose.cache import _video_key
print(_video_key(Path('data/sources/nouveau_chemin/video.mp4')))
"
# Puis, dans ~/.bikefit/cache/landmarks/ :
#   mv video_ANCIENNE_CLE.parquet video_NOUVELLE_CLE.parquet
```

Pour forcer un recalcul volontaire (ex. après une mise à jour de MediaPipe) :
`--no-cache` sur `analyze`/`measure-setup`.

---

## Sources vidéo réseau / webcam

**Symptôme** : `--front` (ou la capture webcam) reste bloqué sans jamais recevoir de
frame.

- **ESP32-CAM** : le firmware ne sert **qu'un client à la fois** — fermer tout onglet
  navigateur ou lecteur VLC déjà ouvert sur `/stream` avant de lancer la capture Python
  (détails : [`firmware/README.md`](../firmware/README.md)).
- **Webcam** : vérifier qu'aucune autre application (Teams, navigateur, une autre
  capture BikeFit) n'a déjà la main sur le périphérique.

**Symptôme** : `Erreur : --front pointe vers la même source que la vidéo de profil`
— garde intentionnelle, `VIDEO` et `--front` ne peuvent pas être la même caméra/fichier ;
utiliser deux fichiers, deux flux, ou deux index webcam distincts (`webcam:0`/`webcam:1`).

---

## `measure-body` : mensurations aberrantes

**Symptôme** : les estimations sont très éloignées de la réalité (facteur ~10).

Vérifier que `taille` dans `config/rider/mensurations.yaml` est bien en
**millimètres** (`1790`, pas `179`) — c'est la seule référence d'échelle utilisée par
`measure-body`, une erreur d'unité s'y fausse silencieusement.

---

## UI Streamlit

**Symptôme** : le formulaire semble figé après avoir cliqué « Lancer l'analyse ».

Normal — l'inférence MediaPipe sur une vidéo de ~60 s prend une à quelques minutes sur
CPU, le spinner reste affiché tout ce temps (pas de retour intermédiaire, contrairement
au CLI qui affiche une progression frame par frame). Uploader deux fois la même vidéo
sans la modifier réutilise le cache (hash de contenu, cf. `ui/helpers.save_upload`) —
le deuxième passage devrait être nettement plus rapide.

**Symptôme** : « Échec de l'analyse — voir le journal » avec un journal peu clair.

Le journal contient la sortie standard/erreur de `cli.analyze` **et**, en cas
d'exception, son type et son message en dernière ligne — chercher l'erreur la plus
proche de la fin du journal.

---

## CI

**Symptôme** : la CI échoue sur `windows-latest` mais pas sur `ubuntu-latest`.

Windows est la plateforme de dev principale et la seule à révéler les bugs
d'encodage console (cp1252, notamment via `typer.echo` avec des caractères
accentués/spéciaux) et de séparateurs de chemin (`\` vs `/`, `pathlib.Path`
partout plutôt que des chaînes concaténées manuellement).
