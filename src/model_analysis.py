import pandas as pd


def get_feature_importance(
    model,
    preprocessor
):
    feature_names = (
        preprocessor.get_feature_names_out()
    )

    feature_names = [
        name.replace("numerical__", "")
        .replace("categorical__", "")
        for name in feature_names
    ]

    importance_table = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance":
                model.feature_importances_
        }
    )

    importance_table = (
        importance_table
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance_table