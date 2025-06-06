# mlflow_integration/main_run.py

import mlflow
import os
import sys

# Add the current directory to the Python path so scripts can be imported
sys.path.append(os.path.dirname(__file__))

import data_preparation
import model_training
import model_scoring

def run_full_pipeline():
    # Set the MLflow experiment name
    mlflow.set_experiment("Housing Price Prediction Full Pipeline")

    # Start the parent MLflow run
    with mlflow.start_run(run_name="Full ML Pipeline Orchestration") as parent_run:
        print(f"Parent Run ID: {parent_run.info.run_id}")
        mlflow.log_param("pipeline_orchestrator_version", "1.0")

        # --- Step 1: Data Preparation (as a nested run) ---
        print("\n--- Running Data Preparation ---")
        with mlflow.start_run(run_name="Data Preparation", nested=True) as child_run_data_prep:
            print(f"Child Run ID (Data Prep): {child_run_data_prep.info.run_id}")
            # Pass parameters to the data preparation function if needed
            data_preparation.run_data_preparation(test_size=0.2, random_state=42)
            print("Data preparation child run completed.")

        # --- Step 2: Model Training (as a nested run) ---
        print("\n--- Running Model Training ---")
        with mlflow.start_run(run_name="Model Training", nested=True) as child_run_model_train:
            print(f"Child Run ID (Model Train): {child_run_model_train.info.run_id}")
            # Pass model training parameters
            trained_model = model_training.run_model_training(n_estimators=100, random_state=42, cv=10)
            print("Model training child run completed.")

        # --- Step 3: Model Scoring (as a nested run) ---
        print("\n--- Running Model Scoring ---")
        with mlflow.start_run(run_name="Model Scoring", nested=True) as child_run_model_score:
            print(f"Child Run ID (Model Score): {child_run_model_score.info.run_id}")
            # Pass the trained model to the scoring function
            if trained_model:
                model_scoring.score_model(trained_model)
            else:
                print("Skipping scoring: Model was not trained successfully.")
            print("Model scoring child run completed.")

        print("\nFull ML Pipeline Orchestration completed.")

if __name__ == "__main__":
    run_full_pipeline()