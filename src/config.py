from pathlib import Path
from datetime import datetime

# Konfiguracja ścieżek
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = MODELS_DIR / "results"

# Stałe konfiguracyjne
CACHE_TTL = 3600  # Czas życia cache w sekundach
DEFAULT_START_DATE = "2000-08-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

# Lista dostępnych metali
AVAILABLE_METALS = ["Złoto", "Srebro", "Platyna", "Pallad", "Miedź"]

# Słownik mapujący nazwy metali na nazwy plików (jeśli są inne)
METAL_MAP = {
    "Złoto": "Gold",
    "Srebro": "Silver",
    "Platyna": "Platinum",
    "Pallad": "Palladium",
    "Miedź": "Copper"
}
