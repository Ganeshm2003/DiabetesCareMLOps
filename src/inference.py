import mlflow
import pandas as pd

def load_model(model_uri: str):
    """
    model_uri examples:
      - "runs:/<run_id>/model"
      - "models:/hospital_readmission/Production"
      - local path like "artifacts/model"
    """
    return mlflow.pyfunc.load_model(model_uri)

def predict_df(model, df: pd.DataFrame):
    preds = model.predict(df)  # this will return probabilities if we save it that way
    return preds
