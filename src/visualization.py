import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def save_confusion_matrix(
    model_name,
    y_true,
    y_prediction,
    output_path
):
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_prediction,
        display_labels=[
            "No early readmission",
            "Readmitted <30 days"
        ],
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def save_roc_curve(
    y_true,
    model_probabilities,
    output_path
):
    figure, axis = plt.subplots()

    for model_name, probabilities in (
        model_probabilities.items()
    ):
        RocCurveDisplay.from_predictions(
            y_true,
            probabilities,
            name=model_name,
            ax=axis
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier"
    )

    axis.set_title("ROC Curve Comparison")
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)


def save_precision_recall_curve(
    y_true,
    model_probabilities,
    output_path
):
    figure, axis = plt.subplots()

    for model_name, probabilities in (
        model_probabilities.items()
    ):
        PrecisionRecallDisplay.from_predictions(
            y_true,
            probabilities,
            name=model_name,
            ax=axis
        )

    axis.set_title(
        "Precision-Recall Curve Comparison"
    )
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)



def save_threshold_plot(
    threshold_table,
    output_path
):
    figure, axis = plt.subplots()

    axis.plot(
        threshold_table["Threshold"],
        threshold_table["Precision"],
        marker="o",
        label="Precision"
    )

    axis.plot(
        threshold_table["Threshold"],
        threshold_table["Recall"],
        marker="o",
        label="Recall"
    )

    axis.plot(
        threshold_table["Threshold"],
        threshold_table["F1_Score"],
        marker="o",
        label="F1-score"
    )

    axis.axvline(
        x=0.45,
        color="gray",
        linestyle="--",
        label="Candidate threshold: 0.45"
    )

    axis.set_title(
        "Random Forest Threshold Analysis"
    )

    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)


def save_feature_importance_plot(
    importance_table,
    output_path,
    number_of_features=15
):
    top_features = (
        importance_table
        .head(number_of_features)
        .sort_values(
            by="Importance",
            ascending=True
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.barh(
        top_features["Feature"],
        top_features["Importance"],
        color="steelblue"
    )

    axis.set_title(
        "Top Random Forest Feature Importances"
    )

    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)