def clean_data(data):
    cleaned_data = data.copy()

    columns_to_drop = [
        "weight",
        "payer_code"
    ]

    cleaned_data = cleaned_data.drop(
        columns=columns_to_drop
    )

    columns_to_fill = [
        "medical_specialty",
        "race",
        "diag_1",
        "diag_2",
        "diag_3"
    ]

    cleaned_data[columns_to_fill] = (
        cleaned_data[columns_to_fill]
        .fillna("Unknown")
    )

    excluded_discharge_codes = [
        11,
        13,
        14,
        19,
        20,
        21
    ]

    cleaned_data = cleaned_data[
        ~cleaned_data[
            "discharge_disposition_id"
        ].isin(excluded_discharge_codes)
    ].copy()

    cleaned_data = cleaned_data.reset_index(
        drop=True
    )

    return cleaned_data