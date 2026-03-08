from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.train import train_model_pipeline
from app.schemas import PredictionInput, PredictionResponse, TrainResponse
from app.predict import predict_model

app = FastAPI(
    title="API de Predição de Defasagem Escolar",
    description="""
API REST para estimar o risco de defasagem escolar de estudantes com base em
atributos acadêmicos e institucionais.

## Funcionalidades
- **/predict**: recebe os dados de um estudante e retorna a predição do modelo
- **/train_model**: executa o pipeline de treinamento e retorna métricas do modelo

## Contexto
Projeto desenvolvido para o Datathon Passos Mágicos com foco em Machine Learning Engineering.
""",
    version="1.0.0",
    contact={
        "name": "Bruno, Vitor, Airton",
    },
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    tags=["Health Check"],
    summary="Verifica se a API está online",
    description="Endpoint simples para validar se a aplicação está ativa.",
)
def health():
    return {"status": "ok"}


@app.post(
    "/predict",
    tags=["Predição"],
    summary="Prediz risco de defasagem escolar",
    description="""
Recebe os dados de um estudante e retorna a predição do modelo treinado.

A resposta inclui:
- classe prevista
- probabilidade estimada
- threshold utilizado
""",
    response_model=PredictionResponse,
    responses={
        200: {
            "description": "Predição realizada com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": 1,
                        "probability": 0.7421,
                        "threshold": 0.47
                    }
                }
            },
        },
        422: {
            "description": "Erro de validação nos dados enviados"
        },
        500: {
            "description": "Erro interno ao realizar a predição"
        },
    },
)
def predict(data: PredictionInput):
    result = predict_model(data)
    return result


@app.post(
    "/train_model",
    tags=["Treinamento"],
    summary="Treina o modelo com os dados disponíveis",
    description="""
Executa o pipeline completo de treinamento do modelo, incluindo:
- pré-processamento
- engenharia de atributos
- oversampling
- treinamento
- ajuste de threshold
- avaliação final

Retorna métricas consolidadas do treinamento.
""",
    response_model=TrainResponse,
    responses={
        200: {
            "description": "Modelo treinado com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "threshold": 0.47,
                        "metrics": {
                            "accuracy": 0.81,
                            "precision": 0.74,
                            "recall": 0.76,
                            "f1_score": 0.75,
                            "roc_auc": 0.87
                        },
                        "confusion_matrix": {
                            "true_negative": 182,
                            "false_positive": 28,
                            "false_negative": 31,
                            "true_positive": 97
                        }
                    }
                }
            },
        },
        500: {
            "description": "Erro interno durante o treinamento"
        },
    },
)
def train_model():
    results = train_model_pipeline()
    return results