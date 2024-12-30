from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

# Inicjalizacja aplikacji FastAPI
app = FastAPI()

# Definicja modelu danych wejściowych
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Dodaj więcej funkcji zgodnie z wymaganiami modelu

# Ścieżka do zapisanego modelu
MODEL_PATH = "/opt/airflow/models/best_model.pkl"

# Ładowanie modelu
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Model nie został znaleziony w {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Wystąpił błąd podczas ładowania modelu: {str(e)}")

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Konwersja danych wejściowych do DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Generowanie przewidywania
        prediction = model.predict(input_data)

        # Zwracanie wyniku jako JSON
        return {"prediction": prediction[0]}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Błąd walidacji danych wejściowych: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Wystąpił błąd podczas generowania przewidywania: {str(e)}")

@app.get("/")
def root():
    return {"message": "API działa! Użyj endpointu /predict, aby uzyskać przewidywania."}
