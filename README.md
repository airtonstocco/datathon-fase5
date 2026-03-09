# Predição de Risco de Defasagem Escolar

**Datathon -- Passos Mágicos**

## 1. Visão Geral do Projeto

### Objetivo

O objetivo deste projeto é desenvolver um modelo de Machine Learning
capaz de **estimar o risco de defasagem escolar de estudantes**
participantes do programa Passos Mágicos.

A identificação antecipada de estudantes em risco permite: - direcionar
intervenções educacionais mais cedo - oferecer suporte pedagógico
específico - melhorar os resultados educacionais do programa

O modelo recebe informações acadêmicas e institucionais do estudante e
retorna a **probabilidade de risco de defasagem escolar**.

------------------------------------------------------------------------

### Solução Proposta

A solução consiste na construção de uma **pipeline completa de Machine
Learning**, contemplando todo o ciclo de vida do modelo:

1.  Pré-processamento dos dados\
2.  Engenharia de atributos\
3.  Balanceamento das classes\
4.  Treinamento de modelos de classificação\
5.  Avaliação do desempenho do modelo\
6.  Ajuste do threshold de decisão\
7.  Serialização do modelo treinado\
8.  Disponibilização da inferência via API REST

O modelo final é exposto por meio de uma **API desenvolvida com
FastAPI**, permitindo que sistemas externos enviem dados e recebam
previsões em tempo real.

------------------------------------------------------------------------

### Stack Tecnológica

**Linguagem** - Python 3.x

**Frameworks de Machine Learning** - scikit-learn - pandas - numpy -
xgboost

**API** - FastAPI

**Serialização** - joblib

**Testes** - pytest

**Empacotamento** - Docker

**Deploy** - Cloud (Render)

**Monitoramento** - logging da aplicação - métricas do modelo

------------------------------------------------------------------------

# 2. Estrutura do Projeto

    datathon/
    │
    ├── app/
    │   ├── main.py
    │   ├── predict.py
    │   └── schemas.py
    │
    ├── src/
    │   ├── train.py
    │   ├── preprocessing.py
    │   ├── feature_engineering.py
    │   ├── evaluate.py
    │   └── utils.py
    │
    ├── model.pkl
    ├── requirements.txt
    ├── Dockerfile
    ├── README.md
    └── base_dados.xlsx

------------------------------------------------------------------------

# 3. Instruções de Deploy

## Pré-requisitos

-   Python 3.12+
-   Docker
-   pip

## Instalação

Criar ambiente virtual:

    python -m venv venv

Ativar ambiente:

Windows

    venv\Scripts\activate

Linux/Mac

    source venv/bin/activate

Instalar dependências:

    pip install -r requirements.txt

------------------------------------------------------------------------

## Executar API localmente

    uvicorn app.main:app --reload

Acesse:

    http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## Executar com Docker

Build da imagem:

    docker build -t datathon-api .

Executar container:

    docker run -p 8000:8000 datathon-api

------------------------------------------------------------------------

# 4. Exemplos de Chamadas à API

## Predição

Endpoint:

    POST /predict

### Exemplo de requisição 1

    curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "atingiu_pv": "Não",
        "cf": 53,
        "cg": 400,
        "ct": 7,
        "inde_22": 6.9,
        "indicado": "Sim",
        "instituicao_de_ensino": "Escola Pública",
        "pedra_20": "Ágata",
        "pedra_21": "Quartzo"
    }'

### Resposta esperada 1

    {
        "prediction": 0,
        "probability": 0.4594,
        "threshold": 0.5511
    }

### Exemplo de requisição 2

    curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "atingiu_pv": "Não",
        "cf": 50,
        "cg": 800,
        "ct": 10,
        "inde_22": 8,
        "indicado": "Não",
        "instituicao_de_ensino": "Escola Pública",
        "pedra_20": "Quartzo",
        "pedra_21": "Quartzo"
    }'

### Resposta esperada 2

    {
        "prediction": 1,
        "probability": 0.5859,
        "threshold": 0.5511
    }

------------------------------------------------------------------------

## Treinamento

Endpoint:

    POST /train_model

Exemplo de resposta:

    {
      "threshold": 0.47,
      "metrics": {
        "accuracy": 0.81,
        "precision": 0.74,
        "recall": 0.76,
        "f1_score": 0.75,
        "roc_auc": 0.87
      }
    }

------------------------------------------------------------------------

# 5. Pipeline de Machine Learning

## Pré-processamento

-   normalização das colunas
-   tratamento de variáveis categóricas
-   criação da variável target

## Engenharia de Features

-   One Hot Encoding
-   seleção de variáveis relevantes

## Balanceamento de Classes

Aplicação de **oversampling** para reduzir desbalanceamento.

## Treinamento e Validação

Modelos testados: - Random Forest - Logistic Regression - XGBoost

Métricas avaliadas: - Accuracy - Precision - Recall - F1-score - ROC-AUC

## Seleção de Modelo

Escolha baseada no melhor equilíbrio entre **precision e recall**.

## Pós-processamento

-   serialização do modelo com joblib
-   exposição via API FastAPI