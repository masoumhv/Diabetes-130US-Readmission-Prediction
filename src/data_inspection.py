
import pandas as pd

def inspect_data(data):
    print("Dataset shape:", data.shape)

    print("\nColumn names:")
    print(data.columns.tolist())

    print("\nData types:")
    print(data.dtypes)

    print("\nReadmission distribution:")
    print(data["readmitted"].value_counts(dropna=False))



def inspect_missing_values(data):
    missing_count = data.isnull().sum()

    missing_percentage = (
        missing_count / len(data) * 100
    ).round(2)

    missing_summary = pd.DataFrame(
        {
            "Missing_Count": missing_count,
            "Missing_Percentage": missing_percentage
        }
    )

    missing_summary = missing_summary[
        missing_summary["Missing_Count"] > 0
    ].sort_values(
        by="Missing_Percentage",
        ascending=False
    )

    print("\nMissing values:")
    print(missing_summary)