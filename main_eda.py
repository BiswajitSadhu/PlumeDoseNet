import config
from src.data_manager import DataManager
from src.visualization import VisualizerEDA
from src.utils import log, ensure_directories

def main():
    viz = VisualizerEDA(output_dir=f"{config.PLOTS_DIR}/eda")
    dm = DataManager()
    
    # 1. Load Raw Data & Analyze
    log("Loading data for EDA...")
    data = "Library/filtered_distance_2000_height_200_train_99.csv"
    data2 = "Library/finer_interpolated_data_filtered_distance_2000_height_200_train_99.csv"
    df_train = pd.read_csv(data)
    df_train2 = pd.read_csv(data2)
    print(f"Low-Res Dataset Shape: {df_train.shape}")
    print(f"High-Res Dataset Shape: {df_train2.shape}")

    df_train[f"{config.TARGET_COL}_log"] = np.log10(df_train[config.TARGET_COL])
    df_train[f"{config.TARGET_COL}_Normalized"] = dm.scaler.fit_transform(df_train[[f"{config.TARGET_COL}_log"]])
    print(f"\nLow-Res Dataset overview after log transforming & Normalizing target col: \n{df_train.head()}")
    print(f"\nLow-Res Dataset Statistical Description: \n{df_train.describe()}")

    df_train2[f"{config.TARGET_COL}_log"] = np.log10(df_train2[config.TARGET_COL])
    df_train2[f"{config.TARGET_COL}_Normalized"] = dm.scaler.fit_transform(df_train2[[f"{config.TARGET_COL}_log"]])
    print(f"\nHigh-Res Dataset overview after log transforming & Normalizing target col: \n{df_train2.head()}")
    print(f"\nHigh-Res Dataset Statistical Description: \n{df_train2.describe()}")

    # 2. Plot Correlations
    log("Generating correlation heatmap of Low-Res Dataset...")
    # Ensure categorical columns are encoded for correlation calculation
    df_encoded = df_train.copy()
    for col in config.CAT_FEATURES:
        df_encoded[col] = dm.label_encoders.get(col, LabelEncoder()).fit_transform(df_encoded[col])

    # Plot Data Corr Heatmap    
    viz.plot_correlation_heatmap(df_encoded, cols=config.NUM_FEATURES + config.CAT_FEATURES, target_col=config.TARGET_COL)

    log("Generating Dose vs Log(Dose) Histplot of Low-Res Dataset...")
    # Log transformed distribution
    viz.plot_distribution_bf_af(df_train[config.TARGET_COL] ,df_train[f'{config.TARGET_COL}_log'])

    log("Generating Low-Res vs High-Res Distance Distribution Historgams..")
    viz.plot_histograms(df_train, df_train2, "Distance (m)")

    log("Generating Category-wise box plots for different Dose scales in Low-Res Data...")
    viz.plot_grouped_boxplots(df_train, "Radionuclide", "Dose", f"{config.TARGET_COL}_Normalized")
    viz.plot_grouped_boxplots(df_train, "Stability Category", "Dose", f"{config.TARGET_COL}_Normalized")

    log("Generating Category-wise Boxen Plots...")
    viz.plot_dose_boxen(df_train, f"{config.TARGET_COL}")

    log("EDA Complete. Check plots/eda folder.")

if __name__ == "__main__":
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    import numpy as np
    main()