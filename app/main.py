"""
Datathon Passos Mágicos - API FastAPI
Autor: Bruno, Vitor, Airton
Ano: 2026

API RESTful para estimar o risco de defasagem escolar de cada estudante
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.train import train_model_pipeline
from app.schemas import PredictionInput
from app.predict import predict_model

app = FastAPI(title="API predição de defasagem escolar")
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# PREVER ALUNOS COM RISCO DE DEFASAGEM
# ===========================
@app.post("/predict")
def predict(data: PredictionInput):
    result = predict_model(data)
    return {"prediction": result}

# ===========================
# TREINAR MODELO COM NOVOS DADOS
# ===========================
@app.post("/train_model")
def train_model():
    results = train_model_pipeline()

    return results