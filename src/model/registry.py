"""MLflow Model Registry integration.

Registers trained models and manages their lifecycle via aliases.

Lifecycle diagram
-----------------

  Every training run produces one new registered version (v1, v2, ...):

      training run ──► register_model() ──► new version (vN)
                                                  │
                              ┌───────────────────┴───────────────────┐
                              │ compare mean_test_rmse with @champion  │
                              └───────────────────┬───────────────────┘
                                    │                         │
                             new RMSE < champion       new RMSE ≥ champion
                                    │                         │
                        old @champion → @challenger     new version → @challenger
                        new version  → @champion        @champion unchanged

Aliases
-------
  @champion   — the version currently recommended for production use.
  @challenger — the most recently trained version that did not beat the
                champion. Useful for A/B testing and rollback analysis.

Loading the champion elsewhere:
    import mlflow.pytorch
    model = mlflow.pytorch.load_model("models:/lstm-stock-predictor@champion")
"""

import logging

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "lstm-stock-predictor"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
METRIC_KEY = "mean_test_rmse"


def _ensure_registered_model(client: MlflowClient) -> None:
    """Create the registered model entity if it does not yet exist."""
    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except MlflowException:
        client.create_registered_model(
            REGISTERED_MODEL_NAME,
            description=(
                "Universal LSTM for multi-asset stock closing price prediction. "
                "Trained on normalised daily Close sequences (window=60 days). "
                "Alias @champion is the production-ready version."
            ),
        )
        logger.info("Created registered model '%s'", REGISTERED_MODEL_NAME)


def register(run_id: str) -> int:
    """Register the model artifact from *run_id* and promote if it is the best.

    Steps:
      1. Fetch ``mean_test_rmse`` logged during that run.
      2. Register the run's ``model`` artifact as a new version.
      3. Tag the version with RMSE, model type, and run ID.
      4. Compare with the current @champion and set aliases accordingly.

    Returns the new version number (int).
    """
    client = MlflowClient()
    _ensure_registered_model(client)

    run = client.get_run(run_id)
    new_rmse = run.data.metrics.get(METRIC_KEY, float("inf"))
    model_type = run.data.params.get("model", "unknown")

    # Register the artifact → assigns a monotonically increasing version number
    mv = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    version = mv.version

    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "rmse", f"{new_rmse:.6f}")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "model_type", model_type)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "run_id", run_id)

    logger.info(
        "Registered '%s' v%s | model_type=%-15s | %s=%.4f",
        REGISTERED_MODEL_NAME, version, model_type, METRIC_KEY, new_rmse,
    )

    # --- Promotion logic ---
    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        champion_rmse = float(champion.tags.get("rmse", float("inf")))

        if new_rmse < champion_rmse:
            logger.info(
                "New model beats champion v%s (%.4f < %.4f) → promoting to @champion",
                champion.version, new_rmse, champion_rmse,
            )
            # Demote the old champion to @challenger before overwriting the alias
            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, champion.version
            )
            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version
            )
        else:
            logger.info(
                "New model does not beat champion v%s (%.4f ≥ %.4f) → tagging as @challenger",
                champion.version, new_rmse, champion_rmse,
            )
            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, version
            )

    except MlflowException:
        # No champion alias exists yet — first registered model becomes champion
        logger.info("No @champion found → v%s becomes the first @champion", version)
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)

    return int(version)


def load_champion() -> object:
    """Load and return the @champion PyTorch model from the registry.

    The model is mapped to the current device (CUDA if available, else CPU)
    so it runs correctly regardless of where it was originally trained.
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
    logger.info("Loading champion model from '%s' → device=%s", uri, device)
    model = mlflow.pytorch.load_model(uri, map_location=device)
    return model.to(device)
