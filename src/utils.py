import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample


def balance_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    diff = np.abs(precisions - recalls)
    best_index = np.argmin(diff)

    return thresholds[best_index]

def oversampling(X_train, y_train):
    train = pd.concat([X_train, y_train], axis=1)
    class_0 = train[train.target == 0]
    class_1 = train[train.target == 1]

    class_0_over = resample(
        class_0,
        replace=True,
        n_samples=len(class_1),
        random_state=42
    )
    train_bal = pd.concat([class_0_over, class_1])
    
    return train_bal