# House Price Prediction — Streamlit App

This project trains an advanced regression model on the Ames Housing dataset and serves the saved model through a Streamlit interface.

## Project structure

- `frontend/app.py` — Streamlit frontend
- `backend/backend.py` — model loading, input preparation, feature engineering, and prediction
- `models/best_model.pkl` — fitted preprocessing + Gradient Boosting pipeline
- `data/processed/cleaned_data.csv` — reference data used for safe defaults and category choices
- `notebooks/` — data preparation, training, and evaluation workflow

## Run locally

Use the same Python environment in which the model was trained:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run frontend/app.py
```

If `python` does not point to your Python 3.13 installation, run:

```powershell
& "C:\Users\ACER\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements.txt
& "C:\Users\ACER\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run frontend/app.py
```

The model file is already a complete sklearn `Pipeline`, so `backend/backend.py` sends raw feature values directly to `best_model.pkl`. The five engineered features used during training are recalculated for every prediction.

