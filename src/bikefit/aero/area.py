"""Surface frontale du cycliste depuis une carte de profondeur (vue de face).

Principe : chaque pixel valide est un point 3D. Un pixel à la profondeur z couvre,
sur un plan perpendiculaire à l'axe optique, (z/fx)·(z/fy) mm². La surface frontale
est la somme de ces aires sur la silhouette du cycliste (vélo inclus), soit
l'approximation de la projection orthographique dans l'axe de déplacement — exacte
pour une caméra centrée sur l'axe du vélo et loin devant la profondeur du corps.

Chaîne par frame :
1. bande de profondeur [near, far] autour du cycliste (le mur du fond tombe dehors) ;
2. retrait du sol (plan estimé une fois par RANSAC : caméra et sol sont fixes) ;
3. plus grande composante connexe = vélo + cycliste ;
4. bouchage des trous **sans mesure** (cuissard noir/brillant qui absorbe l'IR de la
   Kinect) à l'intérieur de la silhouette ; les zones intérieures où l'on voit le fond
   (espace entre bras et buste) restent exclues : l'air y passe ;
5. aire = Σ z²/(fx·fy), en m².

Principe n°1 du projet : jamais de métrique sans sa fiabilité. La part de trous
bouchés (`hole_ratio`) et la part de frames exploitées (`frames_kept`) sont publiées
avec la surface, et la moyenne est faite sur des fenêtres d'une période de pédalage.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel
from scipy import ndimage

from bikefit.aero.recording import DepthIntrinsics, DepthRecording

# Bande de profondeur par défaut autour du buste : ±900 mm englobe la roue avant
# (≈ 0,6-0,8 m devant le buste en position CLM) sans aller chercher le mur du fond.
DEFAULT_BAND_MM = 900.0

# Un point à moins de cette hauteur du plan du sol est considéré comme du sol.
FLOOR_MARGIN_MM = 30.0

# Silhouette en dessous de cette fraction de l'image -> segmentation ratée.
MIN_SILHOUETTE_RATIO = 0.02

# Surface frontale plausible (cycliste + vélo de face). Hors plage : segmentation
# ratée (mur ou sol pris pour le cycliste, cycliste coupé…).
PLAUSIBLE_AREA_M2 = (0.15, 1.2)

# Seuils de fiabilité de la surface frontale.
MAX_RELIABLE_HOLE_RATIO = 0.10
MIN_RELIABLE_FRAMES_KEPT = 0.80
MIN_RELIABLE_WINDOWS = 5

# Une fenêtre (période de pédalage) n'est gardée que si 90 % de ses frames sont valides.
MIN_VALID_FRAMES_PER_WINDOW = 0.90

# Bornes de la période recherchée : 0,25 s = demi-tour à 120 rpm (les deux jambes
# étant symétriques, la silhouette de face peut osciller à 2× la cadence), 1,6 s =
# tour complet à ~38 rpm.
MIN_PERIOD_S = 0.25
MAX_PERIOD_S = 1.6
MIN_AUTOCORR_PEAK = 0.3


class FloorPlane(BaseModel):
    """Plan n·P + d = 0 en mm (repère caméra, y vers le bas), n unitaire orienté vers le
    haut (ny < 0) : n·P + d est alors la hauteur au-dessus du sol."""

    nx: float
    ny: float
    nz: float
    d: float

    def height_mm(self, points: np.ndarray) -> np.ndarray:
        n = np.array([self.nx, self.ny, self.nz])
        return points @ n + self.d  # type: ignore[no-any-return]

    def ray_factor(self, shape: tuple[int, int], intr: DepthIntrinsics) -> np.ndarray:
        """k(u,v) = n·r(u,v), avec r le rayon de profondeur unité : la hauteur d'un pixel
        vaut alors z·k + d. Précalculé une fois -> pas de nuage 3D par frame."""
        h, w = shape
        u = (np.arange(w, dtype=np.float64)[None, :] - intr.cx) / intr.fx
        v = (np.arange(h, dtype=np.float64)[:, None] - intr.cy) / intr.fy
        return self.nx * u + self.ny * v + self.nz  # type: ignore[no-any-return]


class FrontalAreaStats(BaseModel):
    mean_m2: float
    std_m2: float
    min_m2: float
    max_m2: float
    n_windows: int
    window_s: float
    period_detected: bool  # False -> fenêtres d'1 s faute de période de pédalage nette
    hole_ratio: float  # part moyenne de la silhouette sans mesure (bouchée)
    frames_kept: float  # part des frames où la segmentation a réussi
    depth_ref_mm: float  # profondeur de référence (buste)
    near_mm: float
    far_mm: float
    floor_removed: bool

    @property
    def is_measured(self) -> bool:
        return self.n_windows > 0

    @property
    def reliability_issues(self) -> list[str]:
        issues: list[str] = []
        if not self.is_measured:
            return ["aucune fenêtre exploitable"]
        if self.hole_ratio > MAX_RELIABLE_HOLE_RATIO:
            issues.append(
                f"{self.hole_ratio:.0%} de la silhouette sans mesure de profondeur"
                " (vêtement noir/brillant ?)"
            )
        if self.frames_kept < MIN_RELIABLE_FRAMES_KEPT:
            issues.append(f"segmentation réussie sur {self.frames_kept:.0%} des frames seulement")
        if self.n_windows < MIN_RELIABLE_WINDOWS:
            issues.append(f"{self.n_windows} fenêtre(s) seulement (min {MIN_RELIABLE_WINDOWS})")
        lo, hi = PLAUSIBLE_AREA_M2
        if not lo <= self.mean_m2 <= hi:
            issues.append(
                f"surface implausible ({self.mean_m2:.2f} m², attendu {lo}-{hi}) :"
                " vérifier --near-mm/--far-mm et le cadrage"
            )
        if not self.floor_removed:
            issues.append("sol non détecté : il peut être compté dans la surface")
        return issues

    @property
    def is_reliable(self) -> bool:
        return not self.reliability_issues


# ── Géométrie ─────────────────────────────────────────────────────────────────


def depth_to_points(depth_mm: np.ndarray, intr: DepthIntrinsics) -> np.ndarray:
    """Carte de profondeur (H, W) -> nuage (H, W, 3) en mm, repère caméra (y vers le bas)."""
    h, w = depth_mm.shape
    z = depth_mm.astype(np.float64)
    u = np.arange(w, dtype=np.float64)[None, :]
    v = np.arange(h, dtype=np.float64)[:, None]
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=-1)


def median_depth(frames: list[np.ndarray]) -> np.ndarray:
    """Médiane pixel à pixel de plusieurs frames, en ignorant les 0 (pas de mesure)."""
    stack = np.stack(frames).astype(np.float64)
    stack[stack == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # pixel jamais mesuré : "All-NaN slice"
        med = np.nanmedian(stack, axis=0)
    return np.nan_to_num(med, nan=0.0)


def fit_floor_plane(
    depth_mm: np.ndarray,
    intr: DepthIntrinsics,
    iterations: int = 300,
    threshold_mm: float = 20.0,
    min_vertical: float = 0.8,
    min_inlier_ratio: float = 0.2,
    seed: int = 0,
) -> FloorPlane | None:
    """Estime le plan du sol par RANSAC sur la moitié basse de l'image.

    Seuls les plans quasi horizontaux (|ny| ≥ min_vertical, caméra à peu près de
    niveau) sont candidats : un pan du cycliste ou de la roue ne peut pas être pris
    pour le sol. None si aucun plan ne réunit assez de points.
    """
    h, _ = depth_mm.shape
    pts = depth_to_points(depth_mm, intr)[h // 2 :]
    valid = depth_mm[h // 2 :] > 0
    cand = pts[valid]
    if len(cand) < 100:
        return None
    rng = np.random.default_rng(seed)
    if len(cand) > 20_000:
        cand = cand[rng.choice(len(cand), 20_000, replace=False)]

    best_count = 0
    best_inliers: np.ndarray | None = None
    for _ in range(iterations):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if abs(n[1]) < min_vertical:
            continue
        dist = np.abs((cand - p[0]) @ n)
        inliers = dist < threshold_mm
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_inliers is None or best_count < min_inlier_ratio * len(cand):
        return None

    # Affinage moindres carrés (SVD) sur les inliers
    sel = cand[best_inliers]
    centroid = sel.mean(axis=0)
    _, _, vt = np.linalg.svd(sel - centroid, full_matrices=False)
    n = vt[-1]
    if n[1] > 0:  # orienter vers le haut (y image vers le bas)
        n = -n
    if abs(n[1]) < min_vertical:
        return None
    return FloorPlane(nx=float(n[0]), ny=float(n[1]), nz=float(n[2]), d=float(-centroid @ n))


def reference_depth_mm(depth_mm: np.ndarray) -> float | None:
    """Profondeur du buste : médiane des pixels valides de la zone centrale de l'image.

    Le cycliste est cadré au centre (caméra dans l'axe du vélo) ; la fenêtre
    [25 %, 60 %] en hauteur vise le buste et les bras, pas la roue avant.
    Premier quartile plutôt que médiane : si le buste n'est qu'en partie mesuré
    (maillot noir), la médiane basculerait sur le mur du fond, plus loin — et c'est
    le mur entier qui serait « segmenté ». Le cycliste est le plus proche de la fenêtre.
    """
    h, w = depth_mm.shape
    win = depth_mm[int(h * 0.25) : int(h * 0.60), int(w * 0.40) : int(w * 0.60)]
    vals = win[win > 0]
    return float(np.percentile(vals, 25)) if vals.size else None


# ── Segmentation et aire d'une frame ──────────────────────────────────────────


def segment_rider(
    depth_mm: np.ndarray,
    intr: DepthIntrinsics,
    near_mm: float,
    far_mm: float,
    floor: FloorPlane | None = None,
    floor_margin_mm: float = FLOOR_MARGIN_MM,
    floor_factor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Silhouette (vélo + cycliste) et masque des trous bouchés, ou None si échec.

    Renvoie (silhouette, trous) : `trous` ⊂ `silhouette`, pixels sans mesure de
    profondeur entourés par le cycliste, comptés dans la surface.
    """
    import cv2

    valid = depth_mm > 0
    band = valid & (depth_mm >= near_mm) & (depth_mm <= far_mm)
    if floor is not None:
        k = floor_factor if floor_factor is not None else floor.ray_factor(depth_mm.shape, intr)
        band &= depth_mm.astype(np.float64) * k + floor.d > floor_margin_mm

    # Ouverture légère : supprime le moucheté isolé avant l'étiquetage
    band_u8 = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(band_u8, connectivity=8)
    if n_labels <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = labels == biggest
    if component.mean() < MIN_SILHOUETTE_RATIO:
        return None

    # Trous intérieurs : seuls les pixels SANS mesure sont rajoutés. Un trou intérieur
    # où l'on mesure le fond (entre les bras et le buste) est de l'air : il reste dehors.
    holes = _enclosed(component) & ~valid
    return component | holes, holes


def _enclosed(mask: np.ndarray) -> np.ndarray:
    """Pixels hors `mask` mais entièrement entourés par lui (non reliés au bord).

    Remplissage depuis le bord avec OpenCV (floodFill sur une image bordée d'un pixel) :
    même résultat que scipy.ndimage.binary_fill_holes, une dizaine de fois plus rapide
    sur une frame 640×480 — le pipeline traite des milliers de frames.
    """
    import cv2

    h, w = mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:-1, 1:-1] = mask.astype(np.uint8)
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(canvas, flood_mask, (0, 0), 2)
    return canvas[1:-1, 1:-1] == 0


def frame_area_m2(
    depth_mm: np.ndarray,
    silhouette: np.ndarray,
    holes: np.ndarray,
    intr: DepthIntrinsics,
) -> tuple[float, float]:
    """(aire en m², part de la silhouette bouchée). Les trous prennent la profondeur
    du pixel mesuré le plus proche de la silhouette."""
    measured = silhouette & ~holes
    z_sil = depth_mm[silhouette].astype(np.float64)
    if holes.any() and measured.any():
        # Transformée de distance limitée à la boîte englobante (bien plus rapide)
        rows = np.flatnonzero(silhouette.any(axis=1))
        cols = np.flatnonzero(silhouette.any(axis=0))
        box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        z_box = depth_mm[box].astype(np.float64)
        _, (iy, ix) = ndimage.distance_transform_edt(~measured[box], return_indices=True)
        z_box = np.where(holes[box], z_box[iy, ix], z_box)
        z_sil = z_box[silhouette[box]]
    area_mm2 = float(np.sum(z_sil * z_sil)) / (intr.fx * intr.fy)
    hole_ratio = float(holes.sum()) / max(1, int(silhouette.sum()))
    return area_mm2 / 1e6, hole_ratio


# ── Agrégation temporelle ─────────────────────────────────────────────────────


def dominant_period_frames(signal: np.ndarray, fps: float) -> int | None:
    """Période dominante (en frames) d'un signal par autocorrélation, ou None.

    NaN (frames ratées) interpolés avant calcul. La période peut être un demi-cycle
    de pédalage (jambes symétriques) : moyenner sur des fenêtres d'un demi-cycle
    reste non biaisé.
    """
    x = np.asarray(signal, dtype=float)
    ok = ~np.isnan(x)
    if ok.sum() < 4:
        return None
    idx = np.arange(len(x))
    x = np.interp(idx, idx[ok], x[ok])
    x = x - x.mean()
    if np.allclose(x, 0):
        return None

    lo = max(2, int(round(MIN_PERIOD_S * fps)))
    hi = min(len(x) // 2, int(round(MAX_PERIOD_S * fps)))
    if hi <= lo:
        return None
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    ac = ac / ac[0]
    # Normalisation par le recouvrement : sans elle, les grands décalages sont pénalisés
    ac = ac * len(x) / (len(x) - np.arange(len(x)))
    window = ac[lo : hi + 1]
    peak = int(np.argmax(window))
    if window[peak] < MIN_AUTOCORR_PEAK:
        return None
    # Raffinement : éviter de retenir un multiple de la période (2T, 3T…)
    best = lo + peak
    for div in (3, 2):
        sub = int(round(best / div))
        if sub >= lo and ac[sub] >= 0.9 * ac[best]:
            return sub
    return best


def _aggregate(
    areas: np.ndarray,
    holes: np.ndarray,
    fps: float,
    meta: dict[str, float | bool],
) -> FrontalAreaStats:
    ok = ~np.isnan(areas)
    frames_kept = float(ok.mean()) if len(areas) else 0.0
    period = dominant_period_frames(areas, fps)
    detected = period is not None
    win = period if period is not None else max(1, int(round(fps)))

    per_window: list[float] = []
    for start in range(0, len(areas) - win + 1, win):
        chunk = areas[start : start + win]
        valid = ~np.isnan(chunk)
        if valid.mean() >= MIN_VALID_FRAMES_PER_WINDOW:
            per_window.append(float(chunk[valid].mean()))

    vals = np.array(per_window, dtype=float)
    nan = float("nan")
    return FrontalAreaStats(
        mean_m2=round(float(vals.mean()), 4) if vals.size else nan,
        std_m2=round(float(vals.std(ddof=1)), 4) if vals.size > 1 else (0.0 if vals.size else nan),
        min_m2=round(float(vals.min()), 4) if vals.size else nan,
        max_m2=round(float(vals.max()), 4) if vals.size else nan,
        n_windows=int(vals.size),
        window_s=round(win / fps, 3),
        period_detected=detected,
        hole_ratio=round(float(np.nanmean(holes)), 3) if ok.any() else 1.0,
        frames_kept=round(frames_kept, 3),
        depth_ref_mm=float(meta["depth_ref_mm"]),
        near_mm=float(meta["near_mm"]),
        far_mm=float(meta["far_mm"]),
        floor_removed=bool(meta["floor_removed"]),
    )


def analyze_frontal_area(
    recording: DepthRecording,
    near_mm: float | None = None,
    far_mm: float | None = None,
    band_mm: float = DEFAULT_BAND_MM,
    remove_floor: bool = True,
    start_s: float | None = None,
    end_s: float | None = None,
    on_frame: Callable[[int, np.ndarray, np.ndarray | None, np.ndarray | None, float], None]
    | None = None,
) -> tuple[FrontalAreaStats, np.ndarray]:
    """Surface frontale agrégée sur un enregistrement + série par frame (NaN = échec).

    Sol et profondeur de référence sont estimés une fois, sur la médiane de quelques
    frames (caméra et home trainer sont fixes) ; le cycliste qui pédale bouge, le
    reste non. `on_frame(idx, depth, silhouette|None, trous|None, aire)` permet de
    brancher un rendu de debug (aero.render) sans coupler ce module à l'affichage.
    """
    fps = recording.fps
    first = int((start_s or 0.0) * fps)
    last = min(recording.frame_count, int(end_s * fps)) if end_s is not None else None
    last = recording.frame_count if last is None else last
    if last <= first:
        raise ValueError("Intervalle --start/--end vide pour cet enregistrement.")
    indices = range(first, last)

    sample_idx = np.linspace(first, last - 1, num=min(15, last - first)).astype(int)
    background = median_depth([recording.read(int(i)) for i in sample_idx])
    intr = recording.intrinsics

    ref = reference_depth_mm(background)
    if ref is None:
        raise ValueError(
            "Aucune profondeur mesurée au centre de l'image : cycliste hors champ ou"
            " trop près (< 0,8 m pour une Kinect)."
        )
    near = near_mm if near_mm is not None else max(1.0, ref - band_mm)
    far = far_mm if far_mm is not None else ref + band_mm
    floor = fit_floor_plane(background, intr) if remove_floor else None
    floor_factor = floor.ray_factor(background.shape, intr) if floor is not None else None

    areas = np.full(len(indices), np.nan)
    holes = np.full(len(indices), np.nan)
    for k, idx in enumerate(indices):
        depth = recording.read(idx)
        seg = segment_rider(depth, intr, near, far, floor, floor_factor=floor_factor)
        if seg is not None:
            areas[k], holes[k] = frame_area_m2(depth, seg[0], seg[1], intr)
        if on_frame is not None:
            sil, hole_mask = seg if seg is not None else (None, None)
            on_frame(idx, depth, sil, hole_mask, float(areas[k]))

    stats = _aggregate(
        areas,
        holes,
        fps,
        {"depth_ref_mm": ref, "near_mm": near, "far_mm": far, "floor_removed": floor is not None},
    )
    return stats, areas
