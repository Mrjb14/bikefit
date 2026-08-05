from pathlib import Path

# src/bikefit/ → src/ → project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Runtime data lives outside the repo, alongside sessions.db
_USER_DATA = Path.home() / ".bikefit"
MODELS_DIR = _USER_DATA / "models"
CACHE_DIR = _USER_DATA / "cache"

CONFIG_DIR = PROJECT_ROOT / "config"
