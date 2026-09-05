import pandas as pd

from src.data_cleaning import clean_data
from src.data_loading import load_data
from src.data_splitting import split_data
from src.evaluation import analyze_thresholds, evaluate_model
from src.feature_engineering import create_target, select_features
from src.model import train_logistic_regression, train_random_forest
from src.model_analysis import get_feature_importance
from src.preprocessing import create_preprocessor
from src.visualization import (
    save_confusion_matrix,
    save_feature_importance_plot,
    save_precision_recall_curve,
    save_roc_curve,
    save_threshold_plot,
)

DATA_PATH = "data/raw/diabetic_data.csv"
RESULTS_PATH = "results/model_comparison.csv"


def main():
    data = load_data(DATA_PATH)

    cleaned_data = clean_data(data)
    prepared_data = create_target(cleaned_data)

    (
        X,
        y,
        groups,
        numerical_features,
        categorical_features
    ) = select_features(prepared_data)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        _train_groups,
        _test_groups
    ) = split_data(X, y, groups)

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    model_results = []

    logistic_model = train_logistic_regression(
        X_train_processed,
        y_train
    )

    logistic_prediction = logistic_model.predict(
        X_test_processed
    )

    logistic_probability = (
        logistic_model.predict_proba(
            X_test_processed
        )[:, 1]
    )

    logistic_results = evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_prediction,
        logistic_probability
    )

    model_results.append(logistic_results)

    random_forest_model = train_random_forest(
        X_train_processed,
        y_train
    )

    random_forest_prediction = (
        random_forest_model.predict(
            X_test_processed
        )
    )

    random_forest_probability = (
        random_forest_model.predict_proba(
            X_test_processed
        )[:, 1]
    )

    random_forest_results = evaluate_model(
        "Random Forest",
        y_test,
        random_forest_prediction,
        random_forest_probability
    )

    model_results.append(
        random_forest_results
    )

    comparison_table = pd.DataFrame(
        model_results
    )

    comparison_table.to_csv(
        RESULTS_PATH,
        index=False
    )

    print(
        f"\nModel comparison saved to {RESULTS_PATH}"
    )



    save_confusion_matrix(
        "Logistic Regression",
        y_test,
        logistic_prediction,
        "results/logistic_confusion_matrix.png"
    )

    save_confusion_matrix(
        "Random Forest",
        y_test,
        random_forest_prediction,
        "results/random_forest_confusion_matrix.png"
    )

    model_probabilities = {
        "Logistic Regression":
            logistic_probability,
        "Random Forest":
            random_forest_probability
    }

    save_roc_curve(
        y_test,
        model_probabilities,
        "results/roc_curve.png"
    )

    save_precision_recall_curve(
        y_test,
        model_probabilities,
        "results/precision_recall_curve.png"
    )

    threshold_table = analyze_thresholds(
        y_test,
        random_forest_probability
    )

    threshold_table.to_csv(
        "results/random_forest_thresholds.csv",
        index=False
    )

    print("\nRandom Forest threshold analysis:")
    print(threshold_table)


    save_threshold_plot(
        threshold_table,
        "results/random_forest_threshold_analysis.png"
    )

    importance_table = get_feature_importance(
        random_forest_model,
        preprocessor
    )

    importance_table.to_csv(
        "results/random_forest_feature_importance.csv",
        index=False
    )

    save_feature_importance_plot(
        importance_table,
        "results/random_forest_feature_importance.png"
    )

if __name__ == "__main__":
    main()