import os, sys
import time
import config
from src.utils import setup_windows_fix, log, ensure_directories
from src.data_manager import DataManager
from src.models import XGBoostWrapper, RFWrapper, TabNetWrapper
from src.evaluation import calculate_metrics, print_metrics, save_metrics
from src.visualization import Visualizer


def get_user_model_choice():
    """Prompts user to select models to train."""
    print("\n" + "="*30)
    print("   MODEL SELECTION   ")
    print("="*30)
    print("Available models: [tabnet, xgboost, rf]")
    print("Type 'all' to run everything, or separate choices with commas.")
    print("Example: 'xgboost, tabnet'")
    
    choice = input("Enter model(s) to train: ").lower().strip()
    
    available_models = ['tabnet', 'xgboost', 'rf']
    
    selected_models = []
    
    if choice == 'all':
        return available_models
    
    # Parse input string
    parts = [p.strip() for p in choice.split(',')]
    for p in parts:
        if p in available_models:
            selected_models.append(p)
        else:
            log(f"WARNING: Model '{p}' not recognized. Skipping.")
            
    if not selected_models:
        log("No valid models selected. Exiting.")
        sys.exit()
        
    return selected_models

def main():
    metrics_folder = os.path.join(config.SAVE_RESULTS_DIR, "during_train")
    setup_windows_fix()
    ensure_directories([config.OUTPUT_DIR, config.PLOTS_DIR, metrics_folder])
    viz = Visualizer(output_dir=f"{config.PLOTS_DIR}/learning_curves")

    results_list = []

    # 1. User Selection
    model_choices = get_user_model_choice()
    log(f"Selected models: {model_choices}")

    # 2. Data Pipeline
    dm = DataManager(load_artifacts=False) # False = Fit new scalers
    df, X, y_norm = dm.load_and_preprocess(config.DATA_PATH, is_training=True)
    X_train, X_val, X_test, y_train, y_val, y_test = dm.get_train_test_val_split(df, X, y_norm)

    # Dictionary mapping model names to their Class and Save Path To keep loop clean
    model_config_map = {
        'tabnet':  (TabNetWrapper, config.TABNET_MODEL_PATH),
        'xgboost': (XGBoostWrapper, config.XGB_MODEL_PATH),
        'rf':      (RFWrapper, config.RF_MODEL_PATH)
    }

    # 3. Dynamic Training Loop
    for model_name in model_choices:
        log(f"\n--- Starting process for: {model_name.upper()} ---")
        
        WrapperClass, save_path = model_config_map[model_name]

        # Handle specific requirements for TabNet
        if model_name == 'tabnet':
            model = WrapperClass(cat_idxs=dm.cat_idxs, cat_dims=dm.cat_dims)
        else:
            model = WrapperClass()
        
        # Train
        log(f"Training {model_name}...")
        start_time = time.time()
        model.train(X_train, y_train, X_val, y_val)
        log(f"Training finished in {time.time() - start_time:.2f}s")
        
        # Save
        model.save(save_path)
        log(f"Model saved to {save_path}")

        # Evaluate on Test Split
        y_pred_norm = model.predict(X_test)
        
        # Invert scaling
        y_true_orig = dm.inverse_transform_target(y_test)
        y_pred_orig = dm.inverse_transform_target(y_pred_norm)
        
        metrics = calculate_metrics(y_true_orig, y_pred_orig, model_name)
        print_metrics(metrics)
        results_list.append(metrics)

        # 4. Visualization (Conditional)
        if model_name == 'tabnet':
            viz.plot_learning_curves(model.model.history, "TabNet")
        elif model_name == 'xgboost':
            # XGBoostWrapper stores history in self.evals_result
            viz.plot_learning_curves(model.evals_result, "XGBoost")

    # Save all evaluation metrics
    metrics_file = os.path.join(metrics_folder, f"eval_onfly_{time.strftime('%Y_%m%d_%H_%M_%S')}.txt")
    save_metrics(results_list, metrics_file)
    log("All processes completed.")

if __name__ == "__main__":
    main()
