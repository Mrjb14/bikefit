"""Moteur de règles bike fitting CLM/triathlon.

Logique purement déclarative — pas de ML.
Chaque règle mappe un écart angulaire à un réglage et un delta_mm.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from bikefit._paths import CONFIG_DIR
from bikefit.equipment.models import BikeProfile
from bikefit.kinematics.metrics import FitMetrics
from bikefit.recommend.models import FitReport, Priorite, Recommendation

_TARGETS_PATH = CONFIG_DIR / "targets_tt.yaml"


def _load_targets(profile: str) -> dict:
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


def run(
    metrics: FitMetrics,
    bike: BikeProfile,
    profile: str = "triathlon_XL",
) -> FitReport:
    """Génère les recommandations à partir des métriques et du profil vélo."""
    targets, sensitivity = _load_targets(profile)
    recs: list[Recommendation] = []
    hors_cible: list[str] = []
    ok: list[str] = []

    # ── Genou extension PMB → hauteur selle ──────────────────────────────────
    t = targets.get("genou_extension_pmb", {})
    val = metrics.genou_extension_pmb.mean
    t_min, t_max, t_opt = t.get("min", 140), t.get("max", 150), t.get("optimal", 145)
    coeff = sensitivity.get("genou_extension_per_mm_selle_hauteur", 0.65)

    if val > t_max:
        delta_deg = val - t_opt
        delta_mm = _delta_to_mm(delta_deg, coeff)
        butee = (
            bike.plages_reglage.hauteur_selle.min is not None
            and bike.setup_actuel.hauteur_selle_bb is not None
            and bike.setup_actuel.hauteur_selle_bb - (delta_mm or 0)
            < bike.plages_reglage.hauteur_selle.min
        )
        recs.append(Recommendation(
            reglage="hauteur_selle",
            direction="descendre",
            delta_mm=delta_mm,
            motif=(
                f"Genou trop étendu au PMB ({val:.1f}° > {t_max}°). "
                f"La selle est probablement trop haute."
            ),
            priorite=Priorite.haute,
            butee_atteinte=butee,
        ))
        hors_cible.append("genou_extension_pmb")
    elif val < t_min:
        delta_deg = t_opt - val
        delta_mm = _delta_to_mm(delta_deg, coeff)
        recs.append(Recommendation(
            reglage="hauteur_selle",
            direction="monter",
            delta_mm=delta_mm,
            motif=(
                f"Genou trop fléchi au PMB ({val:.1f}° < {t_min}°). "
                f"La selle est probablement trop basse."
            ),
            priorite=Priorite.haute,
        ))
        hors_cible.append("genou_extension_pmb")
    else:
        ok.append("genou_extension_pmb")

    # ── Genou flexion PMH ────────────────────────────────────────────────────
    t = targets.get("genou_flexion_pmh", {})
    val = metrics.genou_flexion_pmh.mean
    t_min, t_max = t.get("min", 65), t.get("max", 78)
    if not (t_min <= val <= t_max):
        direction = "reculer" if val < t_min else "avancer"
        recs.append(Recommendation(
            reglage="recul_selle",
            direction=direction,
            delta_mm=None,
            motif=(
                f"Angle genou PMH hors plage ({val:.1f}°, cible {t_min}-{t_max}°). "
                f"Ajuster le recul de selle."
            ),
            priorite=Priorite.moyenne,
        ))
        hors_cible.append("genou_flexion_pmh")
    else:
        ok.append("genou_flexion_pmh")

    # ── Hanche fermée PMH → hauteur prolongateurs ────────────────────────────
    t = targets.get("hanche_fermee_pmh", {})
    val = metrics.hanche_fermee_pmh.mean
    t_min, t_max, t_opt = t.get("min", 45), t.get("max", 65), t.get("optimal", 55)
    coeff = sensitivity.get("hanche_per_mm_prolong_hauteur", 0.30)

    if val < t_min:
        delta_deg = t_opt - val
        delta_mm = _delta_to_mm(delta_deg, coeff)
        recs.append(Recommendation(
            reglage="hauteur_prolongateurs",
            direction="monter",
            delta_mm=delta_mm,
            motif=(
                f"Hanche trop fermée au PMH ({val:.1f}° < {t_min}°). "
                f"Remonter les pads/prolongateurs pour ouvrir la hanche."
            ),
            priorite=Priorite.moyenne,
        ))
        hors_cible.append("hanche_fermee_pmh")
    elif val > t_max:
        delta_deg = val - t_opt
        delta_mm = _delta_to_mm(delta_deg, coeff)
        recs.append(Recommendation(
            reglage="hauteur_prolongateurs",
            direction="descendre",
            delta_mm=delta_mm,
            motif=(
                f"Hanche trop ouverte au PMH ({val:.1f}° > {t_max}°). "
                f"Descendre les pads pour fermer la hanche."
            ),
            priorite=Priorite.faible,
        ))
        hors_cible.append("hanche_fermee_pmh")
    else:
        ok.append("hanche_fermee_pmh")

    # ── Coude → longueur prolongateurs ──────────────────────────────────────
    t = targets.get("coude", {})
    val = metrics.coude.mean
    t_min, t_max, t_opt = t.get("min", 88), t.get("max", 105), t.get("optimal", 96)
    coeff = sensitivity.get("coude_per_mm_prolong_longueur", 0.25)

    if val > t_max:
        delta_deg = val - t_opt
        delta_mm = _delta_to_mm(delta_deg, coeff)
        recs.append(Recommendation(
            reglage="longueur_prolongateurs",
            direction="raccourcir",
            delta_mm=delta_mm,
            motif=(
                f"Coude trop ouvert ({val:.1f}° > {t_max}°). "
                f"Raccourcir les prolongateurs ou avancer les pads."
            ),
            priorite=Priorite.haute,
        ))
        hors_cible.append("coude")
    elif val < t_min:
        delta_deg = t_opt - val
        delta_mm = _delta_to_mm(delta_deg, coeff)
        recs.append(Recommendation(
            reglage="longueur_prolongateurs",
            direction="allonger",
            delta_mm=delta_mm,
            motif=(
                f"Coude trop fléchi ({val:.1f}° < {t_min}°). "
                f"Allonger les prolongateurs."
            ),
            priorite=Priorite.moyenne,
        ))
        hors_cible.append("coude")
    else:
        ok.append("coude")

    # ── Tronc ────────────────────────────────────────────────────────────────
    t = targets.get("tronc", {})
    val = metrics.tronc.mean
    t_min, t_max = t.get("min", 10), t.get("max", 22)
    if not (t_min <= val <= t_max):
        direction = "monter" if val > t_max else "descendre"
        recs.append(Recommendation(
            reglage="hauteur_prolongateurs",
            direction=direction,
            delta_mm=None,
            motif=(
                f"Angle de tronc hors plage ({val:.1f}°, cible {t_min}-{t_max}°). "
                f"Ajuster la hauteur des pads."
            ),
            priorite=Priorite.faible,
        ))
        hors_cible.append("tronc")
    else:
        ok.append("tronc")

    # ── Cheville PMB ─────────────────────────────────────────────────────────
    t = targets.get("cheville_pmb", {})
    val = metrics.cheville_pmb.mean
    t_min, t_max = t.get("min", 128), t.get("max", 148)
    if not (t_min <= val <= t_max):
        recs.append(Recommendation(
            reglage="cale_chaussure",
            direction="ajuster",
            delta_mm=None,
            motif=(
                f"Cheville hors plage au PMB ({val:.1f}°, cible {t_min}-{t_max}°). "
                f"Vérifier la position de cale / technique de pédalage."
            ),
            priorite=Priorite.faible,
        ))
        hors_cible.append("cheville_pmb")
    else:
        ok.append("cheville_pmb")

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

    return FitReport(
        profil_utilise=profile,
        recommandations=recs,
        angles_hors_cible=hors_cible,
        angles_ok=ok,
        resume=resume,
    )
