import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

def calculate_metrics(y_true, y_pred, model_name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # sMAPE calculation
    smape = np.mean((2 * np.abs(y_pred - y_true)) / (np.abs(y_true) + np.abs(y_pred))) * 100

    results = {
        "Model": model_name,
        "MSE": mse, "RMSE": rmse, "MAE": mae,
        "MAPE": mape, "sMAPE": smape, "R2": r2
    }
    return results

def print_metrics(results):
    print(f"--- {results['Model']} Evaluation ---")
    for k, v in results.items():
        if k != "Model":
            print(f"{k}: {v:.25f}")

def save_metrics(results_list, filepath):
    with open(filepath, "w") as f:
        for result in results_list:
            f.write(f"Model: {result['Model']}\n")
            f.write(f"MSE: {result['MSE']}\n")
            f.write(f"RMSE: {result['RMSE']}\n")
            f.write(f"MAE: {result['MAE']}\n")
            f.write(f"MAPE: {result['MAPE']}%\n")
            f.write(f"sMAPE: {result['sMAPE']}%\n")
            f.write(f"R2: {result['R2']}\n")
            f.write(f"{'-'*50}\n")
    print(f"\n Results saved to: {filepath}")