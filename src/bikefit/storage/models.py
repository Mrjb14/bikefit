"""Modèle de session d'analyse — Phase 3B."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from bikefit.equipment.models import BikeSetup
from bikefit.kinematics.metrics import FitMetrics
from bikefit.recommend.models import FitReport


class Session(BaseModel):
    id: int | None = None
    date: datetime = Field(default_factory=datetime.now)
    video_path: str
    bike_name: str
    profile: str
    setup: BikeSetup = Field(default_factory=BikeSetup)
    metrics: FitMetrics
    report: FitReport
