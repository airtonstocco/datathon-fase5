import numpy as np


if not hasattr(np, "float_"):
    np.float_ = np.float64

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

df = pd.read_excel("base_dados.xlsx")

train = df.sample(frac=0.7, random_state=42)
prod = df.drop(train.index)

report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=train,
    current_data=prod
)

report.save_html("drift_report.html")