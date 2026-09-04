import pandas as pd

from src.data_cleaning import clean_data
from src.feature_engineering import create_target


def test_create_binary_target():
    data = pd.DataFrame(
        {
            "readmitted": [
                "<30",
                ">30",
                "NO"
            ]
        }
    )

    result = create_target(data)

    assert result[
        "readmitted_30_days"
    ].tolist() == [1, 0, 0]


def test_clean_data():
    data = pd.DataFrame(
        {
            "weight": ["?", "?"],
            "payer_code": ["A", "B"],
            "medical_specialty": [
                None,
                "Cardiology"
            ],
            "race": [None, "Caucasian"],
            "diag_1": [None, "250"],
            "diag_2": [None, "401"],
            "diag_3": [None, "428"],
            "discharge_disposition_id": [
                11,
                1
            ]
        }
    )

    result = clean_data(data)

    assert "weight" not in result.columns
    assert "payer_code" not in result.columns

    assert len(result) == 1

    assert (
        result.iloc[0][
            "discharge_disposition_id"
        ]
        == 1
    )