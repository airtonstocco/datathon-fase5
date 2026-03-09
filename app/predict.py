import logging
import joblib
import pandas as pd

from src.preprocessing import normalize_df_columns, transform_instituicao_text
from src.feature_engineering import one_hot_encoding

artifact = joblib.load("model.pkl")
logger = logging.getLogger(__name__)
model = artifact["model"]
model_features = artifact["features"]
best_threshold = artifact.get("threshold", 0.5)


def predict_model(data):
    df = pd.DataFrame([data.model_dump()])


    # se algum campo vier como Enum, converte para string
    for col in df.columns:
        if hasattr(df.loc[0, col], "value"):
            df[col] = df[col].apply(lambda x: x.value)

    df = normalize_df_columns(df)
    df = transform_instituicao_text(df)

    cat_cols = [
        "instituicao_de_ensino",
        "pedra_20",
        "pedra_21",
        "indicado",
        "atingiu_pv",
    ]

    df = one_hot_encoding(df, cat_cols)

    # garante as mesmas colunas do treino
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # remove colunas extras e ordena igual ao treino
    df = df[model_features]

    y_prob = model.predict_proba(df)[:, 1][0]
    y_pred = int(y_prob >= best_threshold)

    logger.info("Nova requisição recebida")

    prediction = model.predict(df)

    logger.info(f"Prediction gerada: {prediction}")
    
    return {
        "prediction": y_pred,
        "probability": round(float(y_prob), 4),
        "threshold": round(float(best_threshold), 4)
    }