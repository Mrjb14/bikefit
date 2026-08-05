from pydantic import BaseModel


class AngleStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    n_cycles: int


class FitMetrics(BaseModel):
    genou_extension_pmb: AngleStats  # max genou par cycle (point mort bas)
    genou_flexion_pmh: AngleStats  # min genou par cycle (point mort haut)
    hanche_fermee_pmh: AngleStats  # min hanche par cycle
    tronc: AngleStats  # moyenne tronc par cycle
    coude: AngleStats  # moyenne coude par cycle
    cheville_pmb: AngleStats  # cheville au PMB
    n_cycles_total: int
    n_cycles_used: int
    detection_rate: float
