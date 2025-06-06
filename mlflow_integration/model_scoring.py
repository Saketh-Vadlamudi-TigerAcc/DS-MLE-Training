# mlflow_integration/model_scoring.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import mlflow
import os

def run_model_scoring():
    # Load prepared test data
    try:
        strat_test_set = pd.read_csv("data/strat_test_set.csv")
    except FileNotFoundError:
        print("Error: Test data not found. Run data_preparation.py first.")
        return

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    # Load the model from MLflow (assuming it was just logged in the parent run)
    # This is a bit tricky with nested runs if you want to load from *another* child run.
    # A common approach is to load from a specific run_id or use a registered model.
    # For simplicity here, we'll assume the model object is passed or just trained.
    # If loaded from MLflow, you'd do:
    # model_uri = f"runs:/{mlflow.active_run().info.run_id}/housing_model" # Or from parent run ID
    # loaded_model = mlflow.sklearn.load_model(model_uri)

    # --- For this example, we'll assume the model is passed directly from main_run.py ---
    # If executed standalone, this part needs a model to load or receive.

    print("Model scoring completed.")


def score_model(model_pipeline):
    # Load prepared test data
    try:
        strat_test_set = pd.read_csv("data/strat_test_set.csv")
    except FileNotFoundError:
        print("Error: Test data not found. Run data_preparation.py first.")
        return

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    # Make predictions
    final_predictions = model_pipeline.predict(X_test)

    # Calculate RMSE
    final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))

    # --- MLflow Tracking ---
    mlflow.log_metric("final_test_rmse", final_rmse)
    print(f"Logged final_test_rmse: {final_rmse:.2f}")

if __name__ == "__main__":
    print("Running model_scoring.py directly. This script typically receives a trained model.")
    # If running directly, you would need to load a model and pass it, e.g.:
    # from mlflow.tracking import MlflowClient
    # client = MlflowClient()
    # # Find a run ID with a logged model from previous steps
    # run_id_with_model = "..." # You'd need to know a run ID that logged a model
    # loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id_with_model}/housing_model")
    # score_model(loaded_model)