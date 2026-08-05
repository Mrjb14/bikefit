"""Tests du moteur de recommandations sur des FitMetrics synthétiques."""

from bikefit.equipment.models import BikeModel, BikeProfile
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
