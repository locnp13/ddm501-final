"""MLflow pyfunc wrapper so the registered model returns P(churn), as the API expects."""
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class ChurnProbaModel(mlflow.pyfunc.PythonModel):
    """Wraps a fitted sklearn Pipeline; `predict` returns the churn probability per row."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        """Return P(churn=1) for each input row."""
        return self.pipeline.predict_proba(model_input)[:, 1]
