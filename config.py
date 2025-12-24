import os

# Paths
DATA_PATH = "Library/finer_interpolated_data_filtered_distance_2000_height_200_train_9975.csv"
OUTPUT_DIR = "output_dir"
SAVE_RESULTS_DIR = os.path.join(OUTPUT_DIR, "saved_metrics")
PLOTS_DIR = "plots"

# Model Paths
XGB_MODEL_PATH = os.path.join(OUTPUT_DIR, "xgboost_model.json")
RF_MODEL_PATH = os.path.join(OUTPUT_DIR, "random_forest.pkl")
TABNET_MODEL_PATH = os.path.join(OUTPUT_DIR, "tabnet_model.pkl")

# Artifact Paths
ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoders.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")

# Features
CAT_FEATURES = ["Radionuclide", "Stability Category"]
NUM_FEATURES = ["Release Height (m)", "Distance (m)"]
TARGET_COL = "Dose"

# Global Params
SEED = 3007
NUM_EPOCHS = 100