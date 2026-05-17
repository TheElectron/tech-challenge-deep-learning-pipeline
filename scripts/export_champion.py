"""Export the @champion model from the MLflow registry to a portable directory.

Run this once before building the Docker image (or whenever the champion changes):

    python scripts/export_champion.py

The exported directory is mounted into the API container via docker-compose.yml
so no MLflow connection is needed at runtime.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import src.config as cfg
from src.model.registry import CHAMPION_ALIAS, REGISTERED_MODEL_NAME

MODELS_DIR = Path("models")
CHAMPION_DIR = MODELS_DIR / "champion"
META_FILE = MODELS_DIR / "champion_meta.json"


def main() -> None:
    conf = cfg.load()
    mlflow.set_tracking_uri(conf["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()

    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"

    print(f"Exporting {uri} (v{mv.version}) → {CHAMPION_DIR} ...")
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(CHAMPION_DIR))

    meta = {
        "name": REGISTERED_MODEL_NAME,
        "version": str(mv.version),
        "alias": CHAMPION_ALIAS,
        "model_type": mv.tags.get("model_type", "unknown"),
        "rmse": float(mv.tags.get("rmse", 0.0)),
    }
    META_FILE.write_text(json.dumps(meta, indent=2))

    print("Export complete:")
    for p in sorted(CHAMPION_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(MODELS_DIR)}")
    print(f"  champion_meta.json")


if __name__ == "__main__":
    main()
