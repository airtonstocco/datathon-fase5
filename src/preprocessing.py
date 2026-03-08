def normalize_df_columns(df):
    df.columns = (
        df.columns
        .str.lower()
        .str.normalize('NFKD')
        .str.encode('ascii', errors='ignore')
        .str.decode('utf-8')
        .str.replace(' ', '_')
    )

    return df

def create_target_column(df):
    df["target"] = (df["defas"] < 0).astype(int)

    return df

def transform_instituicao_text(df):
    df["instituicao_de_ensino"] = (
        df["instituicao_de_ensino"]
        .eq("Escola Pública")
        .map({True: "escola publica", False: "escola privada"})
    )

    return df