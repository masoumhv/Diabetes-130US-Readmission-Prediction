import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model_name,
    y_true,
    y_prediction,
    y_probability
):
    accuracy = accuracy_score(
        y_true,
        y_prediction
    )

    precision = precision_score(
        y_true,
        y_prediction
    )

    recall = recall_score(
        y_true,
        y_prediction
    )

    f1 = f1_score(
        y_true,
        y_prediction
    )

    roc_auc = roc_auc_score(
        y_true,
        y_probability
    )

    matrix = confusion_matrix(
        y_true,
        y_prediction
    )

    tn, fp, fn, tp = matrix.ravel()

    results = {
        "Model": model_name,
        "Accuracy": round(accuracy, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1_Score": round(f1, 3),
        "ROC_AUC": round(roc_auc, 3),
        "True_Negative": int(tn),
        "False_Positive": int(fp),
        "False_Negative": int(fn),
        "True_Positive": int(tp)
    }

    print(f"\n{model_name} evaluation:")

    for metric, value in results.items():
        if metric != "Model":
            print(f"{metric}: {value}")

    print("\nConfusion matrix:")
    print(matrix)

    return results



def analyze_thresholds(
    y_true,
    y_probability
):
    thresholds = [
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70
    ]

    threshold_results = []

    for threshold in thresholds:
        prediction = (
            y_probability >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            prediction,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            prediction,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            prediction,
            zero_division=0
        )

        threshold_results.append(
            {
                "Threshold": threshold,
                "Precision": round(
                    precision,
                    3
                ),
                "Recall": round(
                    recall,
                    3
                ),
                "F1_Score": round(
                    f1,
                    3
                )
            }
        )

    return pd.DataFrame(
        threshold_results
    )