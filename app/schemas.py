from pydantic import BaseModel
from enum import Enum

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
    instituicao_de_ensino: InstituicaoEnsino
    pedra_20: Pedra
    pedra_21: Pedra
    indicado: SimNao
    atingiu_pv: SimNao
    cg: float
    cf: float
    ct: float
    inde_22: float