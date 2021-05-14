import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import cv2
import tifffile

## Basic Data Exploration
BASE_PATH = './Kaggle2/hubmap-kidney-segmentation/'
TRAIN_PATH = os.path.join(BASE_PATH, 'train')

print(os.listdir(BASE_PATH))

df_train = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
print(df_train)

df_sub = pd.read_csv(os.path.join(BASE_PATH, 'sample_submission.csv'))
print(df_sub)

print(f"Number of train images : {df_train.shape[0]}")
print(f"Number of test images : {df_sub.shape[0]}")

df_info = pd.read_csv(os.path.join(BASE_PATH, "HuBMAP-20-dataset_information.csv"))
print(df_info.sample(3))


# Metadata Analysis
print(pd.read_json(os.path.join(BASE_PATH, "train/0486052bb-anatomical-structure.json")))

print(pd.read_json(os.path.join(BASE_PATH, "train/0486052bb.json")))

df_info['split'] = 'test'
df_info.loc[df_info['image_file'].isin(os.listdir(os.path.join(BASE_PATH, 'train'))), "split"] = "train"

df_info["area"] = df_info["width_pixels"] * df_info["height_pixels"]

print(df_info.head())

plt.figure(figsize=(16, 35))
plt.subplot(6, 2, 1)
sn.countplot(x="race", hue="split", data=df_info)
plt.subplot(6, 2, 2)
sn.countplot(x="ethnicity", hue="split", data=df_info)
plt.subplot(6, 2, 3)
sn.countplot(x="sex", hue="split", data=df_info)
plt.subplot(6, 2, 4)
sn.countplot(x="laterality", hue="split", data=df_info)
plt.subplot(6, 2, 5)
sn.histplot(x="age", hue="split", data=df_info)
plt.subplot(6, 2, 6)
sn.histplot(x="weight_kilograms", hue="split", data=df_info)
plt.subplot(6, 2, 7)
sn.histplot(x="height_centimeters", hue="split", data=df_info)
plt.subplot(6, 2, 8)
sn.histplot(x="bmi_kg/m^2", hue="split", data=df_info)
plt.subplot(6, 2, 9)
sn.histplot(x="percent_cortex", hue="split", data=df_info)
plt.subplot(6, 2, 10)
sn.histplot(x="percent_medulla", hue="split", data=df_info)
plt.subplot(6, 2, 11)
sn.histplot(x="area", hue="split", data=df_info)
plt.show()