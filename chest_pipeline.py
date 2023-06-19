from constants import *
import pandas as pd


def rename_columns(df, new_columns):
    return df.rename(columns=new_columns)


def keep_columns(df, columns):
    return df[columns]


def transform_to_dummy(df, column, sep):
    dummies = df[column].str.get_dummies(sep).rename(columns=str.lower)
    df_without_column = df.drop(columns=[column])
    return pd.concat([df_without_column, dummies], axis=1)


def drop_column(df, column):
    return df.drop(columns=[column])


def clean_image_data():
    image_data = pd.read_csv(cxr8_root / "Data_Entry_2017_v2020.csv")
    renaming_transformation = {
        "Image Index": "file_name",
        "Finding Labels": "findings",
        "Patient ID": "patient_id"
    }
    columns_to_keep = [name for name in renaming_transformation.values()]
    cleaned_image_data = (
        image_data.pipe(rename_columns, new_columns=renaming_transformation)
        .pipe(keep_columns, columns=columns_to_keep)
        .pipe(transform_to_dummy, column="findings", sep="|")
        .pipe(drop_column, "no finding")
    )
    chest_image_labels = cleaned_image_data.drop(columns=["patient_id"])
    chest_image_patients = cleaned_image_data[["file_name", "patient_id"]]
    return chest_image_labels, chest_image_patients


def main():
    chest_image_labels, chest_image_patients = clean_image_data()

    chest_image_labels.to_csv(data_root / "chest" / "image_labels.csv", index=False)
    chest_image_patients.to_csv(data_root / "chest" / "image_patients.csv", index=False)


if __name__ == '__main__':
    main()
