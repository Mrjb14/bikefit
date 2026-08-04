from pathlib import Path

# src/bikefit/ → src/ → project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"
CONFIG_DIR = PROJECT_ROOT / "config"
