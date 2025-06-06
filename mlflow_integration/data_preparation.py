# mlflow_integration/data_preparation.py

import pandas as pd
import numpy as np
import os
import tarfile
import urllib.request
from sklearn.model_selection import StratifiedShuffleSplit
import mlflow


def load_housing_data():
    return pd.read_csv("housing.csv")

# --- Main Data Preparation Logic ---
def run_data_preparation(test_size=0.2, random_state=42):
    housing = load_housing_data()

    # Create income categories for stratified sampling
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    # Perform stratified split
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Drop the income_cat column
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # --- MLflow Tracking ---
    mlflow.log_param("data_prep_test_size", test_size)
    mlflow.log_param("data_prep_random_state", random_state)
    mlflow.log_metric("train_set_rows", len(strat_train_set))
    mlflow.log_metric("test_set_rows", len(strat_test_set))

    # Save processed data for downstream tasks
    # IMPORTANT: For simplicity, we'll save to local files for child runs to access.
    # In a real pipeline, you might pass data directly or use a shared storage.
    os.makedirs("data", exist_ok=True)
    strat_train_set.to_csv("data/strat_train_set.csv", index=False)
    strat_test_set.to_csv("data/strat_test_set.csv", index=False)
    mlflow.log_artifact("data/strat_train_set.csv")
    mlflow.log_artifact("data/strat_test_set.csv")

    print("Data preparation completed and data saved to 'data/' folder.")
    return strat_train_set, strat_test_set # Return dataframes for direct use if preferred


if __name__ == "__main__":
    # This block runs only if data_preparation.py is executed directly
    # For nested runs, it will be called by main_run.py
    print("Running data_preparation.py directly (not as a nested run).")
    # To run it as a standalone MLflow run if executed directly:
    # with mlflow.start_run(run_name="Standalone Data Prep"):
    #    run_data_preparation()
    run_data_preparation() # No explicit run here as it will be nested