# mlflow_integration/model_training.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
import mlflow
import os

# --- Custom Transformer (copy from your refactored code) ---
# NOTE: Ensure tr_idx, tb_idx, pop_idx, hh_idx are defined or passed.
# For simplicity, I'm hardcoding them based on standard order if original numerical_features are used.
# In a real project, these would be managed more robustly (e.g., config, or learned from data).
# Assuming numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
TR_IDX = 3 # total_rooms
TB_IDX = 4 # total_bedrooms
POP_IDX = 5 # population
HH_IDX = 6 # households

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True,
                 total_rooms_idx=TR_IDX, households_idx=HH_IDX,
                 population_idx=POP_IDX, total_bedrooms_idx=TB_IDX):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.total_rooms_idx = total_rooms_idx
        self.households_idx = households_idx
        self.population_idx = population_idx
        self.total_bedrooms_idx = total_bedrooms_idx

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, self.total_rooms_idx] / X[:, self.households_idx]
        population_per_household = X[:, self.population_idx] / X[:, self.households_idx]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.total_bedrooms_idx] / X[:, self.total_rooms_idx]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# --- Main Model Training Logic ---
def run_model_training(n_estimators=100, random_state=42, cv=10):
    # Load prepared training data
    try:
        strat_train_set = pd.read_csv("data/strat_train_set.csv")
    except FileNotFoundError:
        print("Error: Training data not found. Run data_preparation.py first.")
        return

    housing_X = strat_train_set.drop("median_house_value", axis=1)
    housing_y = strat_train_set["median_house_value"].copy()

    numerical_features = list(housing_X.drop("ocean_proximity", axis=1).columns)
    categorical_features = ['ocean_proximity']

    # Define pipelines (same as your refactored code)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder(total_rooms_idx=numerical_features.index("total_rooms"),
                                                  households_idx=numerical_features.index("households"),
                                                  population_idx=numerical_features.index("population"),
                                                  total_bedrooms_idx=numerical_features.index("total_bedrooms"))),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    full_preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features),
    ])

    final_ml_pipeline = Pipeline([
        ('preprocessor', full_preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)),
    ])

    # Train the pipeline
    final_ml_pipeline.fit(housing_X, housing_y)

    # --- MLflow Tracking ---
    mlflow.log_param("model_n_estimators", n_estimators)
    mlflow.log_param("model_random_state", random_state)
    mlflow.log_param("model_type", "RandomForestRegressor")

    # Cross-validation
    cv_scores = cross_val_score(final_ml_pipeline, housing_X, housing_y,
                                scoring="neg_mean_squared_error", cv=cv, n_jobs=-1)
    rmse_cv_scores = np.sqrt(-cv_scores)

    mlflow.log_metric("cv_rmse_mean", rmse_cv_scores.mean())
    mlflow.log_metric("cv_rmse_std", rmse_cv_scores.std())

    # Save the trained model as an MLflow artifact
    mlflow.sklearn.log_model(final_ml_pipeline, "housing_model")
    print("Model training completed and model saved to MLflow.")

    return final_ml_pipeline # Return the fitted pipeline for scoring


if __name__ == "__main__":
    print("Running model_training.py directly (not as a nested run).")
    run_model_training() # No explicit run here as it will be nested