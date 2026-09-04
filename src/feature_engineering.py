def create_target(data):
    prepared_data = data.copy()

    prepared_data["readmitted_30_days"] = (
        prepared_data["readmitted"] == "<30"
    ).astype(int)

    return prepared_data

def select_features(data):
    numerical_features = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses"
    ]

    categorical_features = [
        "race",
        "gender",
        "age",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "max_glu_serum",
        "A1Cresult",
        "change",
        "diabetesMed"
    ]

    selected_features = (
        numerical_features + categorical_features
    )

    X = data[selected_features]
    y = data["readmitted_30_days"]
    groups = data["patient_nbr"]

    return (
    X,
    y,
    groups,
    numerical_features,
    categorical_features
)