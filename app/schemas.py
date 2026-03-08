from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class SimNao(str, Enum):
    SIM = "Sim"
    NAO = "Não"


class InstituicaoEnsino(str, Enum):
    PUBLICA = "Escola Pública"
    PRIVADA = "Escola Privada"


class Pedra(str, Enum):
    AGATA = "Ágata"
    AMETISTA = "Ametista"
    QUARTZO = "Quartzo"
    TOPAZIO = "Topázio"


class PredictionInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "instituicao_de_ensino": "Escola Pública",
                "pedra_20": "Ágata",
                "pedra_21": "Quartzo",
                "indicado": "Sim",
                "atingiu_pv": "Não",
                "cg": 400,
                "cf": 53,
                "ct": 7,
                "inde_22": 6.9
            }
        }
    )

    instituicao_de_ensino: InstituicaoEnsino = Field(
        ...,
        description="Tipo de instituição de ensino do estudante."
    )
    pedra_20: Pedra = Field(
        ...,
        description="Classificação pedra referente ao ano 2020."
    )
    pedra_21: Pedra = Field(
        ...,
        description="Classificação pedra referente ao ano 2021."
    )
    indicado: SimNao = Field(
        ...,
        description="Indica se o estudante foi indicado a bolsa."
    )
    atingiu_pv: SimNao = Field(
        ...,
        description="Indica se o estudante atingiu PV."
    )
    cg: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Indicador CG."
    )
    cf: float = Field(
        ...,
        ge=0,
        le=200,
        description="Indicador CF."
    )
    ct: float = Field(
        ...,
        ge=0,
        le=20,
        description="Indicador CT."
    )
    inde_22: float = Field(
        ...,
        ge=0,
        le=10,
        description="INDE do ano de 2022."
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(
        ...,
        description="Classe prevista pelo modelo. Ex.: 0 = sem risco, 1 = com risco."
    )
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probabilidade prevista para a classe positiva."
    )
    threshold: float = Field(
        ...,
        ge=0,
        le=1,
        description="Threshold utilizado para a decisão final."
    )


class MetricsResponse(BaseModel):
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    roc_auc: float = Field(..., ge=0, le=1)


class ConfusionMatrixResponse(BaseModel):
    true_negative: int = Field(..., ge=0)
    false_positive: int = Field(..., ge=0)
    false_negative: int = Field(..., ge=0)
    true_positive: int = Field(..., ge=0)


class TrainResponse(BaseModel):
    threshold: float = Field(
        ...,
        ge=0,
        le=1,
        description="Threshold encontrado durante o treinamento."
    )
    metrics: MetricsResponse
    confusion_matrix: ConfusionMatrixResponse