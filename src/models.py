import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import joblib
import numpy as np
import config

class BaseModel:
    def train(self, X_train, y_train, X_val, y_val): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError
    def get_feature_importance(self, feature_names): raise NotImplementedError

class XGBoostWrapper(BaseModel):
    def __init__(self, params=None):
        self.model = None
        self.params = params if params else {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "max_depth": 30, "eta": 0.05, "subsample": 0.5, "colsample_bytree": 1,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

    def train(self, X_train, y_train, X_val, y_val):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        self.evals_result = {} # Store history
        
        self.model = xgb.train(
            self.params, dtrain, num_boost_round=config.NUM_EPOCHS,
            evals=[(dtrain, "train"), (dval, "eval")], 
            evals_result=self.evals_result,
            early_stopping_rounds=10
        )
        return self.model

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def save(self, path): self.model.save_model(path)
    def load(self, path): 
        self.model = xgb.Booster()
        self.model.load_model(path)

    def get_feature_importance(self, feature_names):
        # XGB returns dict like {'f0': 0.2, 'f1': 0.5...}
        scores = self.model.get_score(importance_type='weight')
        # Map f0 -> Real Name
        imp_dict = {}
        for i, name in enumerate(feature_names):
            imp_dict[name] = scores.get(f'f{i}', 0)
        return imp_dict

class RFWrapper(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'n_estimators': config.NUM_EPOCHS, 
            'max_depth': 15, 
            'n_jobs': -1, 
            'random_state': config.SEED
        }
        if params: default_params.update(params)
        self.model = RandomForestRegressor(**default_params)

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train.ravel())
        return self.model

    def predict(self, X): return self.model.predict(X)
    def save(self, path): joblib.dump(self.model, path, compress=3)
    def load(self, path): self.model = joblib.load(path)

    def get_feature_importance(self, feature_names):
        imps = self.model.feature_importances_
        return dict(zip(feature_names, imps))

class TabNetWrapper(BaseModel):
    def __init__(self, cat_idxs, cat_dims, params=None):
        # Found using optuna hyperparam search
        default_params = {
            'n_d': 16, 'n_a': 16, 'n_steps': 10,
            'gamma': 1.0335564084034523, 'lambda_sparse': 2.73488221131339e-06,
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': {'lr': 0.006284784527220703},
            'scheduler_params': {"mode": "min", "patience": 5, "factor": 0.9},
            'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'device_name': "auto",
        }
        if params: default_params.update(params)
        self.model = TabNetRegressor(cat_idxs=cat_idxs, cat_dims=cat_dims, **default_params)

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train=X_train, y_train=y_train.reshape(-1, 1),
            eval_set=[(X_train, y_train.reshape(-1, 1)), (X_val, y_val.reshape(-1, 1))],
            eval_name=['train', 'val'],
            eval_metric=['rmse'],
            max_epochs=config.NUM_EPOCHS, patience=10,
            batch_size=2048, virtual_batch_size=256
        )
        return self.model

    def predict(self, X): return self.model.predict(X).squeeze()
    def save(self, path): joblib.dump(self.model, path)
    def load(self, path): self.model = joblib.load(path)

    def get_feature_importance(self, feature_names):
        imps = self.model.feature_importances_
        return dict(zip(feature_names, imps))