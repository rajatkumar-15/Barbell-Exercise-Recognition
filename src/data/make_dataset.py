import pandas as pd
from glob import glob
from pathlib import Path
import os
import re

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob(
    r"C:/Users/Lenovo/OneDrive\Desktop/Fitness-Tracker/data/raw/MetaMotion/*.csv"
)

def extract_features_from_filename(file_path):
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    acc_set = 1
    gyro_set = 1

    for f in files:
        f_path = Path(f)
        filename = f_path.name

        participant = filename.split("-")[0]
        label = filename.split("-")[1]
        category_raw = filename.split("-")[2].split("_")[0]
        category = re.sub(r"\d+$", "", category_raw)

        
        df = pd.read_csv(f)
        
        df["participant"] = participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
            
        elif "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_df = pd.concat([gyro_df, df])
            
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyro_df["epoch (ms)"]
    del gyro_df["time (01:00)"]
    del gyro_df["elapsed (s)"]
    
    return acc_df, gyro_df

acc_df, gyro_df = extract_features_from_filename(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:,:3], gyro_df], axis=1)

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "participant",
    "label",
    "category",
    "set",
]
# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyro_x": "mean",
    "gyro_y": "mean",
    "gyro_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

data_merged[:1000].resample("200ms").apply(sampling)

# --------------------------------------------------------------
#split by day
# --------------------------------------------------------------


days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled.info()

data_resampled["set"] = data_resampled["set"].astype("int")

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_resampled.pkl")
data_resampled.to_csv("../../data/interim/01_data_resampled.csv")