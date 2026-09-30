"""Human approval step: `python -m src.training.promote [VERSION]` makes a model the champion."""
import sys

from mlflow.tracking import MlflowClient

from src.training.registry import MODEL_NAME, promote


def main() -> None:
    """Promote the given version (default: current challenger) to champion."""
    version = promote(MlflowClient(), sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"{MODEL_NAME} v{version} is now the champion")


if __name__ == "__main__":
    main()
