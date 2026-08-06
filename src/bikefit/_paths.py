import os
from pathlib import Path

# src/bikefit/ → src/ → project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Runtime data lives outside the repo, alongside sessions.db
_USER_DATA = Path.home() / ".bikefit"
MODELS_DIR = _USER_DATA / "models"
CACHE_DIR = _USER_DATA / "cache"

CONFIG_DIR = PROJECT_ROOT / "config"

# Sorties d'analyse : rapports, métriques, vidéos de debug, images annotées.
# Séparées des sources (data/sources/) pour qu'une analyse n'écrive jamais à côté
# des vidéos d'entrée. Relatif au répertoire courant — le CLI s'utilise depuis la
# racine du projet ; surchargeable par BIKEFIT_OUTPUTS_DIR pour un usage ailleurs.
OUTPUTS_DIR = Path(os.environ.get("BIKEFIT_OUTPUTS_DIR", "data/outputs"))
REPORTS_DIR = OUTPUTS_DIR / "reports"
ANNOTATED_DIR = OUTPUTS_DIR / "annotated"
