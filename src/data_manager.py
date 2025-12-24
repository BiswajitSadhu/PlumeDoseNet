import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils import log
import config

class DataManager:
    def __init__(self, load_artifacts=False):
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        
        # If we are in testing mode, load existing encoders/scalers
        if load_artifacts:
            self._load_artifacts()

    def load_and_preprocess(self, filepath, is_training=True):
        log(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Log Transform Target
        if config.TARGET_COL in df.columns:
            df[f"{config.TARGET_COL}_log"] = np.log10(df[config.TARGET_COL])
        
        # 1. Encode Categoricals
        for col in config.CAT_FEATURES:
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                # Handle unseen labels or strict transform
                df[col] = self.label_encoders[col].transform(df[col])

        # 2. Get TabNet Params
        self.cat_idxs = [i for i, col in enumerate(config.CAT_FEATURES)]
        self.cat_dims = [len(df[col].unique()) for col in config.CAT_FEATURES]

        # 3. Normalize Target (if training)
        target_norm = None
        if is_training:
            target_norm = self.scaler.fit_transform(df[[f"{config.TARGET_COL}_log"]].values.reshape(-1, 1))
            self._save_artifacts()
            
        # 4. Prepare X matrix
        X = np.hstack([df[config.CAT_FEATURES].values, df[config.NUM_FEATURES].values])
        
        return df, X, target_norm

    def get_train_test_val_split(self, df, X, y):
        strat_key = df['Radionuclide'].astype(str) + '_' + df['Stability Category'].astype(str)
        # Split Strat_key as well to match length of next split
        X_train, X_temp, y_train, y_temp, _, strat_temp = train_test_split(X, y, strat_key, test_size=0.2, random_state=config.SEED, stratify=strat_key)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=config.SEED, stratify=strat_temp)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def inverse_transform_target(self, y_pred_norm):
        """Converts normalized predictions back to original Dose scale"""
        # Inverse MinMax -> Inverse Log10
        y_log = self.scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()
        return 10**y_log

    def _save_artifacts(self):
        log("Saving preprocessing artifacts...")
        with open(config.ENCODER_PATH, "wb") as f: pickle.dump(self.label_encoders, f)
        with open(config.SCALER_PATH, "wb") as f: pickle.dump(self.scaler, f)

    def _load_artifacts(self):
        log("Loading preprocessing artifacts...")
        with open(config.ENCODER_PATH, "rb") as f: self.label_encoders = pickle.load(f)
        with open(config.SCALER_PATH, "rb") as f: self.scaler = pickle.load(f)