import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

os.makedirs("Library", exist_ok=True)
Finer_remv_test_path = "Library/finer_interpolated_data_filtered_distance_2000_height_200_test_0025.csv"
Finer_remv_train_path = "Library/finer_interpolated_data_filtered_distance_2000_height_200_train_9975.csv"

df_train = pd.read_csv("Library/filtered_distance_2000_height_200_train_99.csv")
df_finer =  pd.read_csv("Library/finer_interpolated_data_filtered_distance_2000_height_200_train_99.csv")
print(f"Discrete Train Data Shape: df_train={df_train.shape} \t Interpolated Train Data Shape: df_finer={df_finer.shape}")

#Avoid Type Match Errors/warns
for c in ["Release Height (m)", "Distance (m)"]:
    df_train[c] = df_train[c].astype(float)
    df_finer[c] = df_finer[c].astype(float)
    
cols = ["Radionuclide", "Stability Category", "Release Height (m)", "Distance (m)"]

common_rows = pd.merge(df_finer, df_train, on=cols, how='inner')
print(f"The number of similar rows present in df_finer from df_train is: {len(common_rows)}")

# Removing values from df_finer already present in df_train (df_finer_remv = df_finer - df_train)
df_finer_remv = df_finer[~df_finer.set_index(cols).index.isin(df_train.set_index(cols).index)]
print(f"Shape of Interpolated Data after removing samples present in Discrete Data: df_finer_remv={df_finer_remv.shape}")


# Create stratify key to create splits equaly distributed amongst 'Radionuclide' and 'Stability Category'
stratify_key = df_finer_remv['Radionuclide'].astype(str) + '_' + df_finer_remv['Stability Category'].astype(str)

df_finer_remv_train, df_finer_remv_test = train_test_split(
    df_finer_remv, test_size=0.00025, random_state=3007, stratify=stratify_key 
    )
print(f"df_finer_remv Training data shape:{df_finer_remv_train.shape} \ndf_finer_remv Test data shape:{df_finer_remv_test.shape}")

#Save the splits
df_finer_remv_test.to_csv(Finer_remv_test_path, index=False)
df_finer_remv_train.to_csv(Finer_remv_train_path, index=False)
print(f"Interpolated data Training file saved to '{Finer_remv_train_path}' \nInterpolated data Testing file saved to '{Finer_remv_test_path}' ")