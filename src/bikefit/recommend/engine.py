"""Moteur de règles bike fitting CLM/triathlon.

Logique purement déclarative — pas de ML.
Chaque règle mappe un écart angulaire à un réglage et un delta_mm.
"""

from __future__ import annotations

import yaml

from bikefit._paths import CONFIG_DIR
from bikefit.equipment.models import BikeProfile
from bikefit.equipment.rider import Flexibilite, RiderProfile
from bikefit.kinematics.metrics import AngleStats, FitMetrics
from bikefit.recommend.models import FitReport, Priorite, Recommendation

_TARGETS_PATH = CONFIG_DIR / "targets_tt.yaml"


def _load_targets(profile: str) -> tuple[dict, dict]:
    data = yaml.safe_load(_TARGETS_PATH.read_text(encoding="utf-8"))
    if profile not in data:
        available = [k for k in data if not k.startswith("_") and isinstance(data[k], dict)]
        raise ValueError(f"Profil '{profile}' inconnu. Disponibles : {available}")
    return data[profile], data.get("sensitivity", {})


def _delta_to_mm(delta_deg: float, coeff: float) -> float | None:
    """Convertit un écart angulaire en mm via le coefficient de sensibilité."""
    if coeff <= 0:
        return None
    return round(abs(delta_deg) / coeff)


_PRIORITY_RANK = {Priorite.haute: 0, Priorite.moyenne: 1, Priorite.faible: 2}


def _guard_reliability(rec: Recommendation, stat: AngleStats) -> Recommendation:
    """Dégrade une recommandation fondée sur une mesure peu fiable.

    Principe n°1 : jamais de métrique sans sa fiabilité. Une consigne chiffrée en mm
    tirée de points inférés par MediaPipe donnerait une fausse impression de
    précision : on garde la direction (information utile), mais sans delta, en
    priorité faible et avec la raison explicite.
    """
    if stat.is_reliable:
        return rec
    vis = f"visibilité {stat.visibility:.2f}" if stat.visibility is not None else "visibilité ?"
    kept = f"{stat.frames_kept:.0%} des frames" if stat.frames_kept is not None else ""
    detail = ", ".join(x for x in (vis, kept) if x)
    return rec.model_copy(
        update={
            "delta_mm": None,
            "priorite": Priorite.faible,
            "motif": f"[Mesure peu fiable : {detail} — à confirmer] {rec.motif}",
        }
    )


def _merge_same_setting(recs: list[Recommendation]) -> list[Recommendation]:
    """Fusionne les recommandations portant sur un même réglage.

    Hanche (PMH) et tronc agissent tous deux sur la hauteur des prolongateurs :
    sans fusion, le rapport affichait deux consignes (parfois opposées) et le
    schéma du vélo ne gardait que la dernière. Même direction -> une seule consigne
    (priorité la plus haute, plus grand delta chiffré). Directions opposées -> une
    consigne « arbitrer » sans delta, qui expose le conflit au lieu de le masquer.
    """
    by_setting: dict[str, list[Recommendation]] = {}
    for rec in recs:
        by_setting.setdefault(rec.reglage, []).append(rec)

    merged: list[Recommendation] = []
    for group in by_setting.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        best = min(group, key=lambda r: _PRIORITY_RANK[r.priorite])
        motif = " ".join(r.motif for r in group)
        directions = {r.direction for r in group}
        if len(directions) == 1:
            deltas = [r.delta_mm for r in group if r.delta_mm is not None]
            merged.append(
                best.model_copy(
                    update={
                        "delta_mm": max(deltas) if deltas else None,
                        "motif": motif,
                        "butee_atteinte": any(r.butee_atteinte for r in group),
                    }
                )
            )
        else:
            merged.append(
                best.model_copy(
                    update={
                        "direction": "arbitrer",
                        "delta_mm": None,
                        "motif": (
                            f"Indications contradictoires ({' / '.join(sorted(directions))}) : "
                            f"{motif} Privilégier le confort de hanche sur un effort long."
                        ),
                        "butee_atteinte": False,
                    }
                )
            )
    return merged


def _saddle_limit_reached(bike: BikeProfile, delta_mm: float | None, up: bool) -> bool:
    """True si le delta proposé fait sortir la hauteur de selle de sa plage admissible."""
    current = bike.setup_actuel.hauteur_selle_bb
    plage = bike.plages_reglage.hauteur_selle
    if current is None or delta_mm is None:
        return False
    if up:
        return plage.max is not None and current + delta_mm > plage.max
    return plage.min is not None and current - delta_mm < plage.min


def run(
    metrics: FitMetrics,
    bike: BikeProfile,
    profile: str = "triathlon_XL",
    rider: RiderProfile | None = None,
) -> FitReport:
    """Génère les recommandations à partir des métriques, du vélo et du coureur.

    Chaque règle ne s'applique qu'à une métrique effectivement mesurée : une
    métrique NaN (aucun cycle exploitable) part dans `angles_non_mesures` au lieu
    d'être classée « OK » par défaut — une comparaison avec NaN est toujours fausse,
    ce qui la faisait passer pour « dans la plage ».
    """
    targets, sensitivity = _load_targets(profile)
    recs: list[Recommendation] = []
    hors_cible: list[str] = []
    ok: list[str] = []
    non_mesures: list[str] = []

    def measured(key: str, stat: AngleStats) -> bool:
        if stat.is_measured:
            return True
        non_mesures.append(key)
        return False

    def add(rec: Recommendation, stat: AngleStats, key: str) -> None:
        recs.append(_guard_reliability(rec, stat))
        hors_cible.append(key)

    # ── Genou extension PMB → hauteur selle ──────────────────────────────────
    stat = metrics.genou_extension_pmb
    if measured("genou_extension_pmb", stat):
        t = targets.get("genou_extension_pmb", {})
        val = stat.mean
        t_min, t_max, t_opt = t.get("min", 140), t.get("max", 150), t.get("optimal", 145)
        coeff = sensitivity.get("genou_extension_per_mm_selle_hauteur", 0.65)

        if val > t_max or val < t_min:
            trop_haute = val > t_max
            delta_mm = _delta_to_mm(val - t_opt, coeff)
            motif = (
                f"Genou trop étendu au PMB ({val:.1f}° > {t_max}°). "
                "La selle est probablement trop haute."
                if trop_haute
                else f"Genou trop fléchi au PMB ({val:.1f}° < {t_min}°). "
                "La selle est probablement trop basse."
            )
            add(
                Recommendation(
                    reglage="hauteur_selle",
                    direction="descendre" if trop_haute else "monter",
                    delta_mm=delta_mm,
                    motif=motif,
                    priorite=Priorite.haute,
                    butee_atteinte=_saddle_limit_reached(bike, delta_mm, up=not trop_haute),
                ),
                stat,
                "genou_extension_pmb",
            )
        else:
            ok.append("genou_extension_pmb")

    # ── Genou flexion PMH → recul selle ──────────────────────────────────────
    stat = metrics.genou_flexion_pmh
    if measured("genou_flexion_pmh", stat):
        t = targets.get("genou_flexion_pmh", {})
        val = stat.mean
        t_min, t_max = t.get("min", 65), t.get("max", 78)
        if not (t_min <= val <= t_max):
            add(
                Recommendation(
                    reglage="recul_selle",
                    direction="reculer" if val < t_min else "avancer",
                    delta_mm=None,
                    motif=(
                        f"Angle genou PMH hors plage ({val:.1f}°, cible {t_min}-{t_max}°). "
                        "Ajuster le recul de selle."
                    ),
                    priorite=Priorite.moyenne,
                ),
                stat,
                "genou_flexion_pmh",
            )
        else:
            ok.append("genou_flexion_pmh")

    # ── Hanche fermée PMH → hauteur prolongateurs ────────────────────────────
    stat = metrics.hanche_fermee_pmh
    if measured("hanche_fermee_pmh", stat):
        t = targets.get("hanche_fermee_pmh", {})
        val = stat.mean
        t_min, t_max, t_opt = t.get("min", 45), t.get("max", 65), t.get("optimal", 55)
        coeff = sensitivity.get("hanche_per_mm_prolong_hauteur", 0.30)

        if val < t_min:
            add(
                Recommendation(
                    reglage="hauteur_prolongateurs",
                    direction="monter",
                    delta_mm=_delta_to_mm(t_opt - val, coeff),
                    motif=(
                        f"Hanche trop fermée au PMH ({val:.1f}° < {t_min}°). "
                        "Remonter les pads/prolongateurs pour ouvrir la hanche."
                    ),
                    priorite=Priorite.moyenne,
                ),
                stat,
                "hanche_fermee_pmh",
            )
        elif val > t_max:
            add(
                Recommendation(
                    reglage="hauteur_prolongateurs",
                    direction="descendre",
                    delta_mm=_delta_to_mm(val - t_opt, coeff),
                    motif=(
                        f"Hanche trop ouverte au PMH ({val:.1f}° > {t_max}°). "
                        "Descendre les pads pour fermer la hanche."
                    ),
                    priorite=Priorite.faible,
                ),
                stat,
                "hanche_fermee_pmh",
            )
        else:
            ok.append("hanche_fermee_pmh")

    # ── Coude → longueur prolongateurs ──────────────────────────────────────
    stat = metrics.coude
    if measured("coude", stat):
        t = targets.get("coude", {})
        val = stat.mean
        t_min, t_max, t_opt = t.get("min", 88), t.get("max", 105), t.get("optimal", 96)
        coeff = sensitivity.get("coude_per_mm_prolong_longueur", 0.25)

        if val > t_max:
            add(
                Recommendation(
                    reglage="longueur_prolongateurs",
                    direction="raccourcir",
                    delta_mm=_delta_to_mm(val - t_opt, coeff),
                    motif=(
                        f"Coude trop ouvert ({val:.1f}° > {t_max}°). "
                        "Raccourcir les prolongateurs ou avancer les pads."
                    ),
                    priorite=Priorite.haute,
                ),
                stat,
                "coude",
            )
        elif val < t_min:
            add(
                Recommendation(
                    reglage="longueur_prolongateurs",
                    direction="allonger",
                    delta_mm=_delta_to_mm(t_opt - val, coeff),
                    motif=f"Coude trop fléchi ({val:.1f}° < {t_min}°). Allonger les prolongateurs.",
                    priorite=Priorite.moyenne,
                ),
                stat,
                "coude",
            )
        else:
            ok.append("coude")

    # ── Tronc → hauteur prolongateurs ────────────────────────────────────────
    # Angle mesuré par rapport à l'horizontale : plus il est grand, plus le buste
    # est redressé. Trop redressé -> DESCENDRE les pads ; trop à plat -> les MONTER.
    # (Le sens était inversé jusqu'ici, et n'était couvert par aucun test.)
    stat = metrics.tronc
    if measured("tronc", stat):
        t = targets.get("tronc", {})
        val = stat.mean
        t_min, t_max = t.get("min", 10), t.get("max", 22)

        # Ajustement selon la flexibilité du coureur
        flex_note = ""
        if rider is not None and rider.profil.flexibilite is not None:
            flex = rider.profil.flexibilite
            if flex == Flexibilite.faible:
                t_min = min(t_min + 5, t_max - 2)
                flex_note = " (plancher relevé : flexibilité faible)"
            elif flex == Flexibilite.excellente:
                t_min = max(t_min - 3, 5)
                flex_note = " (plancher abaissé : excellente flexibilité)"

        if not (t_min <= val <= t_max):
            trop_redresse = val > t_max
            etat = "trop redressé" if trop_redresse else "trop à plat"
            add(
                Recommendation(
                    reglage="hauteur_prolongateurs",
                    direction="descendre" if trop_redresse else "monter",
                    delta_mm=None,
                    motif=(
                        f"Tronc {etat} ({val:.1f}°, cible {t_min}-{t_max}°{flex_note}). "
                        f"{'Descendre' if trop_redresse else 'Monter'} les pads."
                    ),
                    priorite=Priorite.faible,
                ),
                stat,
                "tronc",
            )
        else:
            ok.append("tronc")

    # ── Cheville PMB → cale ──────────────────────────────────────────────────
    stat = metrics.cheville_pmb
    if measured("cheville_pmb", stat):
        t = targets.get("cheville_pmb", {})
        val = stat.mean
        t_min, t_max = t.get("min", 128), t.get("max", 148)
        if not (t_min <= val <= t_max):
            add(
                Recommendation(
                    reglage="cale_chaussure",
                    direction="ajuster",
                    delta_mm=None,
                    motif=(
                        f"Cheville hors plage au PMB ({val:.1f}°, cible {t_min}-{t_max}°). "
                        "Vérifier la position de cale / technique de pédalage."
                    ),
                    priorite=Priorite.faible,
                ),
                stat,
                "cheville_pmb",
            )
        else:
            ok.append("cheville_pmb")

    # ── Largeur coudes → largeur des pads (plan frontal, vue de face) ────────
    # Silencieux tant que la mesure n'est pas exploitable : il faut la vue de face
    # ET la calibration ArUco de la caméra de face (sinon la largeur reste en
    # pixels) ET une mesure fiable (les coudes peuvent être masqués) ET la largeur
    # d'épaules du coureur, puisque la cible est un ratio. Mieux vaut aucune
    # recommandation qu'une recommandation fondée sur une mesure devinée.
    largeur = metrics.largeur_coudes
    epaules = rider.mensurations.largeur_epaules if rider is not None else None

    if largeur is not None and largeur.is_reliable and epaules:
        t = targets.get("largeur_coudes_ratio", {})
        r_min, r_max = t.get("min", 0.80), t.get("max", 1.00)
        r_opt = t.get("optimal", 0.90)
        ratio = largeur.mean / epaules
        cible_mm = r_opt * epaules

        if ratio > r_max or ratio < r_min:
            trop_large = ratio > r_max
            delta_total = abs(largeur.mean - cible_mm)
            add(
                Recommendation(
                    reglage="largeur_pads",
                    direction="resserrer" if trop_large else "elargir",
                    delta_mm=round(delta_total),
                    motif=(
                        f"Écart des coudes {'trop large' if trop_large else 'trop étroit'} "
                        f"({largeur.mean:.0f} mm = {ratio:.0%} de la largeur d'épaules, "
                        f"cible {r_min:.0%}-{r_max:.0%}). "
                        f"Viser ~{cible_mm:.0f} mm, soit {delta_total / 2:.0f} mm par pad."
                    ),
                    priorite=Priorite.faible,
                ),
                largeur,
                "largeur_coudes",
            )
        else:
            ok.append("largeur_coudes")

    recs = _merge_same_setting(recs)

    # ── Résumé ───────────────────────────────────────────────────────────────
    n_haute = sum(1 for r in recs if r.priorite == Priorite.haute)
    if not recs:
        resume = "Position dans les plages cibles — aucun réglage urgent."
    elif n_haute > 0:
        resume = (
            f"{n_haute} réglage(s) prioritaire(s) identifié(s) sur {len(recs)} total. "
            f"Commencer par : {', '.join(r.reglage for r in recs if r.priorite == Priorite.haute)}."
        )
    else:
        resume = f"{len(recs)} réglage(s) suggéré(s), aucun urgent."

    if non_mesures:
        resume += f" | Non mesuré(s) : {', '.join(non_mesures)} — vérifier le cadrage."

    # Contexte coureur — notes informatives
    rider_notes: list[str] = []
    if rider is not None:
        selle_geo = rider.saddle_height_geo_mm(bike.longueur_manivelle_mm)
        if selle_geo is not None:
            rider_notes.append(f"Référence géométrique hauteur selle (LeMond) : ~{selle_geo} mm")
        duree = rider.objectif.duree_effort_min
        if duree is not None and duree >= 240:
            rider_notes.append(
                f"Effort long ({duree} min) — favoriser la conservation musculaire"
                " sur l'agressivité aérodynamique."
            )
        if rider.objectif.type_epreuve is not None:
            epreuve_profile = rider.objectif.type_epreuve
            if epreuve_profile != profile:
                rider_notes.append(
                    f"Profil utilisé '{profile}' diffère du type d'épreuve cible"
                    f" '{epreuve_profile}' dans les mensurations."
                )

    if rider_notes:
        resume = resume + " | " + " | ".join(rider_notes)

    return FitReport(
        profil_utilise=profile,
        recommandations=recs,
        angles_hors_cible=hors_cible,
        angles_ok=ok,
        angles_non_mesures=non_mesures,
        resume=resume,
    )
