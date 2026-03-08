from src.preprocessing import normalize_df_columns, transform_instituicao_text, create_target_column
from src.feature_engineering import one_hot_encoding
from src.utils import oversampling, balance_threshold
from src.evaluate import evaluate_model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model_pipeline():

    df = pd.read_excel("./base_dados.xlsx")
    df = normalize_df_columns(df)
    df = create_target_column(df)
    df = transform_instituicao_text(df)

    # Selecionadas variáveis categóricas com grande diferença de % entre as classes
    cat_cols = [
        "instituicao_de_ensino",
        "pedra_20",
        "pedra_21",
        "pedra_22",
        "indicado",
        "atingiu_pv",
    ]
    # Selecionadas variáveis numéricas com correlação maior ou igual a 0.3
    num_cols = [    
        "cg",
        "cf",
        "ct",
        "inde_22"
    ]
    features_selection = cat_cols + num_cols + ["target"]
    df = df[features_selection]

    # One Hot Encoding em variáveis categóricas
    df = one_hot_encoding(df, cat_cols)

    # Separar features e target
    features = df.columns.tolist()
    features.remove("target")
    X = df[features]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Oversampling feito somente na amostra de treino
    train_bal = oversampling(X_train, y_train)
    X_train_bal = train_bal[features]
    y_train_bal = train_bal["target"]

    # Treino com o modelo que apresentou o desempenho mais balanceado
    model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
    )
    model.fit(X_train_bal, y_train_bal)

    # Avaliação do modelo
    y_prob = model.predict_proba(X_test)[:, 1]
    best_threshold = balance_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_threshold).astype(int)
    evaluation = evaluate_model(best_threshold, y_test, y_pred, y_prob)

    # Salvar modelo
    joblib.dump(
        {
            "model": model,
            "threshold": best_threshold,
            "features": features
        },
        "model.pkl"
    )
    
    return {
        "status": "Modelo treinado com sucesso",
        "evaluation": evaluation
    }