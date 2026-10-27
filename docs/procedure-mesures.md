# Procédure de prise de mesures

Ce document décrit comment remplir les deux fichiers de configuration :
- `config/rider/mensurations.yaml` — mensurations corporelles
- `config/bikes/giant_trinity_2019.yaml` — réglages actuels du vélo (`setup_actuel`)

> Ces deux fichiers sont gitignorés (données personnelles). Sur un clone neuf, partir des modèles
> versionnés dans `config/examples/` :
> `cp config/examples/giant_trinity_2019.example.yaml config/bikes/giant_trinity_2019.yaml` et
> `cp config/examples/mensurations.example.yaml config/rider/mensurations.yaml`.

Chaque mesure peut être obtenue **automatiquement** (photo ou vidéo) puis **validée** au mètre ruban.
La validation physique est obligatoire avant de considérer une mesure comme fiable.

---

## 1. Mensurations corporelles

### Option A — Photo automatique (`measure-body`)

**Conditions de prise de vue :**

- Vêtements moulants (cuissard + maillot) pour que les articulations soient visibles.
- Fond uni, éclairage diffus (pas de contre-jour).
- Appareil horizontal, cadré de la tête aux pieds avec un peu de marge (10-15 cm de chaque côté).
- Tenir une posture anatomique naturelle : dos droit, bras légèrement écartés, pieds à largeur d'épaules.
- Vue de **face** privilégiée (largeur épaules + hanches plus précises). La vue de profil est aussi acceptée.

**Procédure :**

```bash
# Sans écriture — vérification visuelle d'abord
python -m bikefit measure-body data/sources/photo_debout.jpg

# Si les valeurs sont cohérentes, écriture dans le YAML
python -m bikefit measure-body data/sources/photo_debout.jpg --write
```

La photo annotée (squelette + crochets de cotation) est écrite dans
`data/outputs/annotated/<photo>_annotated.jpg` — à regarder avant tout `--write`.

La commande utilise `taille` (renseignée en **mm**, ex. 1790 pour 1,79 m) pour calibrer l'échelle.
Elle ne remplace **jamais** un champ déjà renseigné — seuls les champs `null` sont mis à jour.

⚠️ Toutes les longueurs de `config/rider/mensurations.yaml` sont en **millimètres**. Saisir 179
au lieu de 1790 fausse silencieusement toute la mise à l'échelle.

**Précision attendue :** ±5 à 10 % — suffisant comme point de départ, à confirmer au ruban.

---

### Option B — Mètre ruban (mesure de référence)

Mesurer pieds nus, debout sur une surface plane. Faire mesurer par une deuxième personne pour les mesures du dos.

| Champ YAML          | Ce qu'on mesure                                            | Repères anatomiques                          |
|---------------------|------------------------------------------------------------|----------------------------------------------|
| `taille`            | Hauteur totale debout                                      | Sol → sommet du crâne (tête droite, regard horizontal) |
| `entrejambe`        | Hauteur pubis                                              | Sol → pubis, pieds nus côte à côte. Variante : livre pressé contre le pubis, mesurer jusqu'au sol |
| `longueur_femur`    | Hanche → genou                                             | Grand trochanter → interligne articulaire du genou (face externe) |
| `longueur_tibia`    | Genou → cheville                                           | Interligne genou → malléole externe           |
| `longueur_buste`    | Pubis → C7                                                 | Pubis → vertèbre C7 (bosse nuque, tête penchée en avant) |
| `longueur_bras`     | Épaule → poignet                                           | Acromion → styloïde ulnaire (poignet côté auriculaire), bras le long du corps |
| `largeur_epaules`   | Largeur biacromiale                                        | Acromion droit → acromion gauche (saillies osseuses sommets des épaules) |
| `largeur_hanches`   | Largeur biiliaque                                          | EIAS droite → EIAS gauche (épines iliaques antéro-supérieures — saillies osseuses du bassin) |
| `pointure`          | Pointure chaussure de cyclisme                             | Taille EU — chaussure portée sur le vélo      |

**Vérifications croisées utiles :**

- `entrejambe` ≈ 0,47 × `taille` (±3 cm pour un adulte standard)
- `longueur_femur` + `longueur_tibia` ≈ `entrejambe` − 30 à 40 mm (cheville au-dessus du sol)
- `longueur_buste` + `entrejambe` ≈ `taille` − `longueur_tête` (≈ 23 cm pour 179 cm)

---

## 2. Réglages actuels du vélo (`setup_actuel`)

> **Prérequis :** le vélo est sur le Tacx Neo 2, posture normale, selle et prolongateurs à leur position habituelle.

### Option A — Vidéo automatique (`measure-setup`)

**Pré-requis :**

1. Imprimer et coller le marqueur ArUco `docs/aruco_marker_100mm.png` (ID=0) sur le tube de selle ou
   le diagonal, côté caméra. La face imprimée doit être parfaitement plane (pas de plis).
2. Vérifier que le marqueur mesure exactement **100 mm de côté** à l'impression (règle, pas « ajuster
   à la page »).
3. **Recommandé (Phase 3A+)** : imprimer et coller un second marqueur ArUco `docs/aruco_marker_id1_100mm.png`
   (ID=1) directement sur le boîtier de pédalier (BB), bien plan et visible caméra. S'il est détecté de
   façon fiable, `measure-setup` l'utilise comme centre BB exact (±5 mm visé) au lieu d'estimer sa
   position par fit de cercle sur la trajectoire de cheville (±20 mm, toujours utilisé en repli
   automatique si ce marqueur est absent, mal collé ou masqué).

**Conditions de tournage :**

- Caméra perpendiculaire au plan de pédalage (vue strictement de profil).
- Côté filmé visible : genou, cheville, hanche, poignet, les deux marqueurs ArUco dans le même plan.
- Durée : au moins 60 secondes à rythme de pédalage stable (70–90 rpm).
- La totalité du cycle de la cheville doit être visible (ne pas zoomer trop près).
- Le marqueur BB (ID=1) doit rester visible sur la majorité de la vidéo — il sera occasionnellement
  masqué par le pied/la manivelle en passant, ce qui est normal (position médiane sur les frames où il
  est détecté).

**Procédure :**

```bash
# Sans écriture — vérifier les estimations
python -m bikefit measure-setup data/sources/video.mp4 --marker-size 100 --side right

# Si cohérent, écriture dans le YAML
python -m bikefit measure-setup data/sources/video.mp4 --marker-size 100 --side right --write

# Avec découpage (ignorer montée/descente, comme --start/--end sur analyze) — recommandé,
# la montée/descente peut faire perdre le marqueur BB ou fausser le fit de cercle
python -m bikefit measure-setup data/sources/video.mp4 --marker-size 100 --side right --start 20 --end 70 --write
```

La commande recherche automatiquement le marqueur BB (ID=1) et affiche la méthode utilisée :

```
Recherche du marqueur BB (ID=1)...
  Marqueur BB détecté (58/60 frames) -> BB localisé précisément (±5 mm).
```

ou, en son absence :

```
Recherche du marqueur BB (ID=1)...
  Marqueur BB (ID=1) non détecté ou peu fiable -> BB estimé par fit de cercle sur la trajectoire de cheville (±20 mm).
```

La frame annotée (trajectoire de la cheville, BB estimé, cotes) est écrite dans
`data/outputs/annotated/<video>_annotated.jpg`.

**Correction obligatoire pour `hauteur_selle_bb` :**
La commande mesure la distance BB → **articulation de la hanche** au PMB, pas jusqu'au sommet de la selle.
L'écart hanche → selle est d'environ **30 à 50 mm** selon la morphologie et le cuissard.

```
hauteur_selle_bb_réelle ≈ hauteur_selle_bb_estimée − 40 mm  (valeur typique)
```

Confirmer avec la mesure ruban ci-dessous.

**Précision attendue :** ±5 mm avec marqueur BB détecté de façon fiable, ±20 mm en repli (fit de
cercle) — dans les deux cas, validation ruban obligatoire avant de considérer la mesure comme fiable.

**Dépannage — pointage manuel du BB (`--bb-px`) :**
Si le marqueur ID=1 n'est pas détecté de façon fiable (mal collé, masqué trop souvent, mauvais
éclairage) et que le fit de cercle (±20 mm) est insuffisant, il est possible de pointer le pixel du
BB à la main (résolution vidéo originale, caméra fixe obligatoire) :

```bash
python -m bikefit measure-setup data/sources/video.mp4 --marker-size 100 --side right --bb-px 812,540
```

`--bb-px` est prioritaire sur le marqueur ArUco et le fit de cercle — précision indicative ±10 mm
(dépend de la précision du pointage). Utile aussi pour valider/comparer les deux autres méthodes sur
une même vidéo.

---

### Option B — Mètre ruban (mesure de référence)

Mesurer sur le vélo monté sur le home trainer, selle et guidon en position de roulage.

| Champ YAML              | Ce qu'on mesure                                              | Méthode                                      |
|-------------------------|--------------------------------------------------------------|----------------------------------------------|
| `hauteur_selle_bb`      | Axe boîtier de pédalier → sommet de la selle                | Ruban le long du tube de selle, dans l'axe du tube (pas vertical) |
| `recul_selle`           | Recul horizontal de la selle par rapport au BB               | Fil à plomb depuis l'axe du BB, mesurer jusqu'au nez de selle (ou centre des rails) |
| `hauteur_guidon_bb`     | Axe BB → surface supérieure des pads repose-mains            | Ruban vertical de l'axe BB jusqu'au pad       |
| `reach_prolongateurs`   | Axe BB → extrémité des prolongateurs (là où reposent les mains) | Ruban horizontal de l'axe BB jusqu'à l'extrémité des extensions |

> **Conseil :** marquer l'axe du BB avec un adhésif temporaire (pièce de 1 € collée sur le boîtier) pour servir de repère visuel lors des mesures.

**Signe de `hauteur_guidon_bb` :**
- Positif si les pads sont plus hauts que le BB (rare en CLM)
- Négatif si les pads sont plus bas que le BB (typique en triathlon/CLM)

---

## 3. Plages de réglage mécaniques (`plages_reglage`)

Ces butées définissent jusqu'où les réglages peuvent physiquement aller.

| Champ YAML                   | Ce qu'on mesure                                            |
|------------------------------|------------------------------------------------------------|
| `hauteur_selle.min`          | `hauteur_selle_bb` avec la selle au plus bas mécaniquement  |
| `hauteur_selle.max`          | `hauteur_selle_bb` avec la selle au plus haut (marque de sécurité sur la tige) |
| `recul_selle.min/max`        | Positions extrêmes des rails sur le chariot de selle         |
| `reach_prolong.min/max`      | Longueur minimale et maximale des extensions réglables       |

Remplir ces champs permet au moteur de recommandation de signaler quand une recommendation dépasse une butée mécanique.

---

## 4. Boucle de validation complète

Le cycle de validation est le suivant :

```
1. measure-body (photo) → mensurations.yaml   [bootstrap]
2. Confirmer au mètre ruban → corriger si écart > 10 mm
3. Filmer sur le home trainer (sans marqueur)
   → python -m bikefit data/sources/video.mp4 --start 20 --end 70
4. Lire les recommandations (delta_mm) → appliquer physiquement le réglage
5. Mesurer le nouveau setup au ruban → remplir setup_actuel
6. Refilmer (avec marqueur ArUco)
   → python -m bikefit data/sources/video.mp4 --marker-size 100 --start 20 --end 70 --save-session
7. Comparer les sessions → python -m bikefit compare ID_avant ID_après
8. Répéter jusqu'à ce que les angles soient dans les cibles
```

**Toujours découper** (`--start`/`--end`) : la montée et la descente du vélo faussent lourdement
les moyennes. Sur une capture de 79 s, les ~30 s hors pédalage faisaient passer la recommandation
de prolongateurs de −48 mm à −111 mm.

**Vérifier la fiabilité** : si l'analyse affiche `métriques peu fiables (points inférés)`, la
mesure concernée repose sur des positions devinées par MediaPipe — corriger le cadrage ou
l'éclairage de la caméra avant d'exploiter ce chiffre.

**Critère de succès Phase 3A+ :** mesure ArUco (`measure-setup`) ≈ mesure ruban à ± 5 mm — implémenté
(marqueur BB ID=1) mais **pas encore validé sur capture réelle** ; à confirmer une fois le marqueur
BB filmé et comparé au ruban.

---

## Récapitulatif des commandes

```bash
# Mensurations corporelles depuis une photo debout
python -m bikefit measure-body photo.jpg [--write]

# Setup vélo depuis une vidéo avec marqueur ArUco (BB via ID=1 si présent, sinon fit de cercle)
python -m bikefit measure-setup video.mp4 --marker-size 100 [--side right] [--write]

# Analyse complète + recommandations
python -m bikefit video.mp4 --side right --profile triathlon_XL --save-session

# Historique des sessions
python -m bikefit history

# Comparaison avant/après réglage
python -m bikefit compare ID1 ID2
```
