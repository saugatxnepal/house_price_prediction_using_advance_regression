"""Prediction service for the Streamlit house-price application.

The serialized best_model.pkl is a complete sklearn Pipeline: it already
contains both the fitted preprocessor and the Gradient Boosting regressor.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
TARGET = "SalePrice"

ENGINEERED_FEATURES = {
    "TotalBathrooms",
    "TotalSF",
    "TotalPorchSF",
    "HouseAge",
    "RemodelAge",
}


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained sklearn pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run notebooks/05_model_training.ipynb first."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_reference_data() -> pd.DataFrame:
    """Load the cleaned Ames data used to create sensible form defaults."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Reference data not found at {DATA_PATH}.")
    return pd.read_csv(DATA_PATH)


def model_features() -> list[str]:
    model = load_model()
    features = getattr(model, "feature_names_in_", None)
    if features is None:
        raise ValueError("The saved pipeline does not expose feature_names_in_.")
    return [str(feature) for feature in features]


@lru_cache(maxsize=1)
def default_values() -> dict[str, Any]:
    """Return robust defaults for every raw input expected by the model."""
    data = load_reference_data()
    defaults: dict[str, Any] = {}

    for column in model_features():
        if column in ENGINEERED_FEATURES:
            continue
        if column not in data.columns:
            raise ValueError(f"Training data is missing model feature: {column}")

        series = data[column]
        if pd.api.types.is_numeric_dtype(series):
            defaults[column] = float(series.median())
        else:
            modes = series.dropna().mode()
            defaults[column] = str(modes.iloc[0]) if not modes.empty else "None"

    return defaults


def category_options(column: str) -> list[str]:
    """Return fitted encoder categories, with the default listed first.

    The fitted encoder is authoritative: it returns exactly the category
    values the saved pipeline can recognize, including values produced after
    the categorical imputer ran during model training.
    """
    options: list[str] = []
    preprocessor = load_model().named_steps.get("preprocessor")
    for name, transformer, columns in getattr(preprocessor, "transformers_", []):
        if name != "cat" or column not in columns:
            continue
        encoder = transformer.named_steps.get("encoder")
        column_index = list(columns).index(column)
        options = sorted(str(value) for value in encoder.categories_[column_index])
        break

    if not options:
        data = load_reference_data()
        if column not in data.columns:
            return []
        options = sorted(str(value) for value in data[column].dropna().unique())

    default = str(default_values().get(column, ""))
    if default in options:
        options.remove(default)
        options.insert(0, default)
    return options


def build_model_input(values: Mapping[str, Any]) -> pd.DataFrame:
    """Build the one-row, feature-engineered DataFrame expected by the pipeline."""
    row = default_values().copy()
    row.update(values)

    year_sold = float(row["YrSold"])
    year_built = float(row["YearBuilt"])
    year_remodeled = float(row["YearRemodAdd"])
    if year_built > year_sold:
        raise ValueError("Year built cannot be later than the year sold.")
    if year_remodeled < year_built or year_remodeled > year_sold:
        raise ValueError("Remodel year must be between the build year and sale year.")

    row["TotalBathrooms"] = (
        float(row["FullBath"])
        + 0.5 * float(row["HalfBath"])
        + float(row["BsmtFullBath"])
        + 0.5 * float(row["BsmtHalfBath"])
    )
    row["TotalSF"] = (
        float(row["TotalBsmtSF"])
        + float(row["1stFlrSF"])
        + float(row["2ndFlrSF"])
    )
    row["TotalPorchSF"] = (
        float(row["OpenPorchSF"])
        + float(row["EnclosedPorch"])
        + float(row["3SsnPorch"])
        + float(row["ScreenPorch"])
    )
    row["HouseAge"] = year_sold - year_built
    row["RemodelAge"] = year_sold - year_remodeled

    ordered = {feature: row[feature] for feature in model_features()}
    return pd.DataFrame([ordered])


def predict_price(values: Mapping[str, Any]) -> float:
    """Predict a sale price in US dollars."""
    prediction = float(load_model().predict(build_model_input(values))[0])
    if not np.isfinite(prediction):
        raise ValueError("The model returned a non-finite prediction.")
    return max(0.0, prediction)


def market_percentile(price: float) -> float:
    """Compare a prediction with prices in the training data."""
    prices = load_reference_data()[TARGET].dropna().to_numpy()
    return float((prices <= price).mean() * 100)


def reference_price_stats() -> dict[str, float]:
    prices = load_reference_data()[TARGET].dropna()
    return {
        "median": float(prices.median()),
        "minimum": float(prices.min()),
        "maximum": float(prices.max()),
    }

