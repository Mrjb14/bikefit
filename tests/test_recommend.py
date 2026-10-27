"""Tests du moteur de recommandations sur des FitMetrics synthétiques."""

import pytest

from bikefit.equipment.models import BikeModel, BikeProfile
from bikefit.equipment.rider import (
    Mensurations,
    ObjectifEpreuve,
    ProfilCoureur,
    RiderProfile,
)
from bikefit.kinematics.metrics import AngleStats, FitMetrics
from bikefit.recommend import recommend
from bikefit.recommend.models import Priorite


def _stats(mean: float, std: float = 1.0, n: int = 10) -> AngleStats:
    return AngleStats(mean=mean, std=std, min=mean - 5, max=mean + 5, n_cycles=n)


def _metrics(**overrides: float) -> FitMetrics:
    defaults = dict(
        genou_extension_pmb=145.0,
        genou_flexion_pmh=71.0,
        hanche_fermee_pmh=55.0,
        tronc=16.0,
        coude=96.0,
        cheville_pmb=138.0,
    )
    defaults.update(overrides)
    return FitMetrics(
        genou_extension_pmb=_stats(defaults["genou_extension_pmb"]),
        genou_flexion_pmh=_stats(defaults["genou_flexion_pmh"]),
        hanche_fermee_pmh=_stats(defaults["hanche_fermee_pmh"]),
        tronc=_stats(defaults["tronc"]),
        coude=_stats(defaults["coude"]),
        cheville_pmb=_stats(defaults["cheville_pmb"]),
        n_cycles_total=10,
        n_cycles_used=10,
        detection_rate=1.0,
    )


def _bike() -> BikeProfile:
    return BikeProfile(modele=BikeModel(marque="Giant", nom="Trinity", annee=2019, taille="M"))


class TestRecommendOk:
    def test_all_in_range_no_recs(self) -> None:
        report = recommend(_metrics(), _bike(), profile="triathlon_XL")
        assert report.recommandations == []
        assert report.angles_hors_cible == []
        assert len(report.angles_ok) == 6

    def test_resume_ok(self) -> None:
        report = recommend(_metrics(), _bike(), profile="triathlon_XL")
        assert "aucun" in report.resume.lower()


class TestRecommendGenou:
    def test_genou_trop_haut_selle_a_descendre(self) -> None:
        report = recommend(_metrics(genou_extension_pmb=152.0), _bike(), profile="triathlon_XL")
        rec = next(r for r in report.recommandations if r.reglage == "hauteur_selle")
        assert rec.direction == "descendre"
        assert rec.priorite == Priorite.haute
        assert rec.delta_mm is not None and rec.delta_mm > 0

    def test_genou_trop_bas_selle_a_monter(self) -> None:
        report = recommend(_metrics(genou_extension_pmb=135.0), _bike(), profile="triathlon_XL")
        rec = next(r for r in report.recommandations if r.reglage == "hauteur_selle")
        assert rec.direction == "monter"

    def test_delta_mm_proportionnel(self) -> None:
        report_petit = recommend(
            _metrics(genou_extension_pmb=151.0), _bike(), profile="triathlon_XL"
        )
        report_grand = recommend(
            _metrics(genou_extension_pmb=158.0), _bike(), profile="triathlon_XL"
        )
        delta_petit = next(
            r for r in report_petit.recommandations if r.reglage == "hauteur_selle"
        ).delta_mm
        delta_grand = next(
            r for r in report_grand.recommandations if r.reglage == "hauteur_selle"
        ).delta_mm
        assert delta_grand > delta_petit  # type: ignore[operator]


class TestRecommendCoude:
    def test_coude_ouvert_raccourcir(self) -> None:
        report = recommend(_metrics(coude=116.0), _bike(), profile="triathlon_XL")
        rec = next(r for r in report.recommandations if r.reglage == "longueur_prolongateurs")
        assert rec.direction == "raccourcir"
        assert rec.priorite == Priorite.haute

    def test_coude_ok_pas_de_rec(self) -> None:
        report = recommend(_metrics(coude=96.0), _bike(), profile="triathlon_XL")
        assert not any(r.reglage == "longueur_prolongateurs" for r in report.recommandations)


class TestRecommendHanche:
    def test_hanche_trop_fermee_monter_prolong(self) -> None:
        report = recommend(_metrics(hanche_fermee_pmh=40.0), _bike(), profile="triathlon_XL")
        rec = next(r for r in report.recommandations if r.reglage == "hauteur_prolongateurs")
        assert rec.direction == "monter"

    def test_priorite_moyenne(self) -> None:
        report = recommend(_metrics(hanche_fermee_pmh=40.0), _bike(), profile="triathlon_XL")
        rec = next(r for r in report.recommandations if r.reglage == "hauteur_prolongateurs")
        assert rec.priorite == Priorite.moyenne


class TestRecommendResume:
    def test_resume_mentionne_prioritaires(self) -> None:
        report = recommend(
            _metrics(genou_extension_pmb=155.0, coude=120.0), _bike(), profile="triathlon_XL"
        )
        assert "prioritaire" in report.resume.lower()


# ── Largeur des pads (plan frontal) ──────────────────────────────────────────


class TestLargeurPads:
    """La règle ne doit parler que si la mesure est réellement exploitable.

    Elle exige la vue de face + la calibration ArUco (sinon largeur_coudes est None),
    une mesure fiable (les coudes peuvent être inférés) et la largeur d'épaules du
    coureur, puisque la cible est un ratio et non une valeur absolue.
    """

    @staticmethod
    def _wide_stats(mean: float, *, reliable: bool = True) -> AngleStats:
        return AngleStats(
            mean=mean,
            std=3.0,
            min=mean - 5,
            max=mean + 5,
            n_cycles=20,
            visibility=0.9 if reliable else 0.3,
            frames_kept=0.95 if reliable else 0.1,
        )

    def _run(self, largeur: AngleStats | None, epaules: int | None = 400):
        metrics = _metrics().model_copy(update={"largeur_coudes": largeur})
        rider = RiderProfile(
            mensurations=Mensurations(largeur_epaules=epaules),
            profil=ProfilCoureur(),
            objectif=ObjectifEpreuve(),
        )
        return recommend(metrics, _bike(), profile="triathlon_XL", rider=rider)

    def _pads(self, report):
        return [r for r in report.recommandations if r.reglage == "largeur_pads"]

    def test_within_target_gives_no_recommendation(self) -> None:
        # 360 / 400 = 90 % -> dans la cible 80-100 %
        report = self._run(self._wide_stats(360.0))
        assert self._pads(report) == []
        assert "largeur_coudes" in report.angles_ok

    def test_too_wide_suggests_narrowing(self) -> None:
        # 440 / 400 = 110 % -> au-dessus du max (100 %)
        report = self._run(self._wide_stats(440.0))
        rec = self._pads(report)[0]
        assert rec.direction == "resserrer"
        assert rec.delta_mm == pytest.approx(80, abs=1)  # cible 0.90*400 = 360
        assert "par pad" in rec.motif

    def test_too_narrow_suggests_widening(self) -> None:
        # 280 / 400 = 70 % -> sous le min (80 %)
        report = self._run(self._wide_stats(280.0))
        rec = self._pads(report)[0]
        assert rec.direction == "elargir"
        assert rec.delta_mm == pytest.approx(80, abs=1)

    def test_silent_without_calibration(self) -> None:
        # Pas d'ArUco -> largeur_coudes reste None
        report = self._run(None)
        assert self._pads(report) == []
        assert "largeur_coudes" not in report.angles_ok

    def test_silent_when_measurement_unreliable(self) -> None:
        # Coudes inférés : aucune recommandation plutôt qu'une reco devinée
        report = self._run(self._wide_stats(440.0, reliable=False))
        assert self._pads(report) == []

    def test_silent_without_shoulder_width(self) -> None:
        # La cible est un ratio : sans largeur d'épaules, rien à comparer
        report = self._run(self._wide_stats(440.0), epaules=None)
        assert self._pads(report) == []


# ── Régressions de l'audit (tronc, NaN, fusion, fiabilité, butées) ────────────


def _nan_stats() -> AngleStats:
    nan = float("nan")
    return AngleStats(mean=nan, std=nan, min=nan, max=nan, n_cycles=0)


class TestRecommendTronc:
    """Tronc mesuré par rapport à l'horizontale : grand = redressé."""

    def test_trunk_too_upright_lowers_pads(self) -> None:
        report = recommend(_metrics(tronc=30.0), _bike(), profile="triathlon_XL")
        rec = next(r for r in report.recommandations if r.reglage == "hauteur_prolongateurs")
        assert rec.direction == "descendre"

    def test_trunk_too_flat_raises_pads(self) -> None:
        report = recommend(_metrics(tronc=4.0), _bike(), profile="triathlon_XL")
        rec = next(r for r in report.recommandations if r.reglage == "hauteur_prolongateurs")
        assert rec.direction == "monter"


class TestRecommendMergeSameSetting:
    def test_same_direction_merged_into_one(self) -> None:
        # Hanche trop fermée (monter) + tronc trop à plat (monter) -> une seule consigne
        report = recommend(
            _metrics(hanche_fermee_pmh=40.0, tronc=4.0), _bike(), profile="triathlon_XL"
        )
        recs = [r for r in report.recommandations if r.reglage == "hauteur_prolongateurs"]
        assert len(recs) == 1
        assert recs[0].direction == "monter"
        assert recs[0].delta_mm is not None  # le delta chiffré de la règle hanche est gardé
        assert recs[0].priorite == Priorite.moyenne
        assert {"hanche_fermee_pmh", "tronc"} <= set(report.angles_hors_cible)

    def test_opposite_directions_flagged_as_conflict(self) -> None:
        # Hanche trop fermée (monter) mais tronc trop redressé (descendre)
        report = recommend(
            _metrics(hanche_fermee_pmh=40.0, tronc=30.0), _bike(), profile="triathlon_XL"
        )
        recs = [r for r in report.recommandations if r.reglage == "hauteur_prolongateurs"]
        assert len(recs) == 1
        assert recs[0].direction == "arbitrer"
        assert recs[0].delta_mm is None
        assert "contradictoires" in recs[0].motif


class TestRecommendNotMeasured:
    def test_nan_metric_is_not_reported_ok(self) -> None:
        metrics = _metrics().model_copy(update={"coude": _nan_stats()})
        report = recommend(metrics, _bike(), profile="triathlon_XL")
        assert "coude" not in report.angles_ok
        assert "coude" not in report.angles_hors_cible
        assert report.angles_non_mesures == ["coude"]
        assert "Non mesuré" in report.resume

    def test_nan_metric_produces_no_recommendation(self) -> None:
        metrics = _metrics().model_copy(update={"genou_flexion_pmh": _nan_stats()})
        report = recommend(metrics, _bike(), profile="triathlon_XL")
        assert all(r.reglage != "recul_selle" for r in report.recommandations)


class TestRecommendUnreliable:
    def test_unreliable_metric_downgraded_without_delta(self) -> None:
        shaky = AngleStats(
            mean=160.0, std=1.0, min=155, max=165, n_cycles=10, visibility=0.3, frames_kept=0.2
        )
        metrics = _metrics().model_copy(update={"genou_extension_pmb": shaky})
        report = recommend(metrics, _bike(), profile="triathlon_XL")
        rec = next(r for r in report.recommandations if r.reglage == "hauteur_selle")
        assert rec.direction == "descendre"  # la direction reste informative
        assert rec.delta_mm is None  # mais pas de chiffre trompeur
        assert rec.priorite == Priorite.faible
        assert "peu fiable" in rec.motif


class TestRecommendSaddleLimits:
    def _bike_with_range(self, current: float, lo: float, hi: float) -> BikeProfile:
        return BikeProfile.model_validate(
            {
                "modele": {"marque": "Giant", "nom": "Trinity", "annee": 2019, "taille": "M"},
                "setup_actuel": {"hauteur_selle_bb": current},
                "plages_reglage": {"hauteur_selle": {"min": lo, "max": hi}},
            }
        )

    def test_raise_beyond_max_flags_limit(self) -> None:
        # Genou très fléchi -> monter de plusieurs cm, alors que la tige est presque en butée
        report = recommend(
            _metrics(genou_extension_pmb=125.0),
            self._bike_with_range(current=745, lo=700, hi=750),
            profile="triathlon_XL",
        )
        rec = next(r for r in report.recommandations if r.reglage == "hauteur_selle")
        assert rec.direction == "monter"
        assert rec.butee_atteinte is True

    def test_lower_within_range_no_limit(self) -> None:
        report = recommend(
            _metrics(genou_extension_pmb=152.0),
            self._bike_with_range(current=740, lo=700, hi=750),
            profile="triathlon_XL",
        )
        rec = next(r for r in report.recommandations if r.reglage == "hauteur_selle")
        assert rec.butee_atteinte is False
