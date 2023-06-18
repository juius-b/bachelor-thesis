from pathlib import Path
import pandas as pd

fp_root = Path("/mnt/jbrockma/")
cxr8_root = fp_root / "CXR8"
cxr8_images_root = cxr8_root / "images" / "images"
chest_root = fp_root / "bachelor-thesis-images" / "chest"


def rename_columns(df, transformation):
    return df.rename(columns=transformation)


def keep_columns(df, columns):
    return df[columns]


def dummyfy_column(df, column, sep):
    dummies = df[column].str.get_dummies(sep).rename(columns=str.lower)
    df_without_column = df.drop(column, axis=1)
    return pd.concat([df_without_column, dummies], axis=1)


def drop_column(df, column):
    return df.drop(column, axis=1)


def normalize_chest_data(df):
    transformation = {"Image Index": "file_name", "Finding Labels": "findings"}
    normalized_chest_image_data = (
        df.pipe(rename_columns, transformation=transformation)
        .pipe(keep_columns, columns=["file_name", "findings"])
        .pipe(dummyfy_column, column="findings", sep="|")
        .pipe(drop_column, "no finding")
    )
    return normalized_chest_image_data


def main():
    df = pd.read_csv(cxr8_root / "Data_Entry_2017_v2020.csv")

    normalized_chest_image_data = normalize_chest_data(df)

    normalized_chest_image_data.to_csv(fp_root / "bachelor-thesis-data" / "chest-image-data.csv", index=False)


if __name__ == '__main__':
    main()
