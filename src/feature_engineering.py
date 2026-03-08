import pandas as pd

def one_hot_encoding(df, cat_cols):
    df = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True
    )

    return df