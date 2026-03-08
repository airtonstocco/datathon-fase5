from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(best_threshold, y_test, y_pred, y_prob):
    print(f"Threshold equilibrado: {best_threshold:.3f}")

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"Acurácia: {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"ROC AUC: {roc:.3f}")
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("\nMatriz de Confusão:")
    print("                Pred 0    Pred 1")
    print(f"Real 0 (neg)     {tn:5d}      {fp:5d}")
    print(f"Real 1 (pos)     {fn:5d}      {tp:5d}")
    
    return {
        "threshold": round(best_threshold, 3),
        "metrics": {
            "accuracy": round(acc, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "roc_auc": round(roc, 3)
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    }