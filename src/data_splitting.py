from sklearn.model_selection import StratifiedGroupKFold


def split_data(X, y, groups):
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_index, test_index = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    train_groups = groups.iloc[train_index]
    test_groups = groups.iloc[test_index]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_groups,
        test_groups
    )