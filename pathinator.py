from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
dataset_folder = DATASET_DIR / "training_dataset"
dataset_files = [f for f in Path(dataset_folder).iterdir() if f.is_file()]


models_folder = BASE_DIR / "models"