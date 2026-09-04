import pandas as pd

from src.data_splitting import split_data


def test_no_patient_overlap():
    number_of_samples = 50

    X = pd.DataFrame(
        {
            "feature": range(
                number_of_samples
            )
        }
    )

    y = pd.Series(
        [0, 1] * 25
    )

    groups = pd.Series(
        range(number_of_samples)
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_groups,
        test_groups
    ) = split_data(X, y, groups)

    shared_patients = (
        set(train_groups)
        & set(test_groups)
    )

    assert len(shared_patients) == 0

    assert (
        len(X_train) + len(X_test)
        == number_of_samples
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)