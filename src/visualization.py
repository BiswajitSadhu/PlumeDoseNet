import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import config
from src.utils import log
from sklearn.metrics import r2_score, mean_absolute_percentage_error

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        # Apply the user's specific theme
        sns.set_theme(
            style="whitegrid",
            rc={
                'font.family': ['DejaVu Serif'], 'font.size': 20,
                'axes.titlesize': 20, 'axes.labelsize': 18,
                'xtick.labelsize': 15, 'ytick.labelsize': 15,
                'legend.fontsize': 15, 'axes.linewidth': 2
            }
        )
        os.makedirs(output_dir, exist_ok=True)

    def plot_learning_curves(self, history, model_name):
        """Plots training vs validation loss over epochs."""
        log(f"Plotting learning curves for {model_name}...")
        plt.figure(figsize=(10, 6))
        
        if model_name == "TabNet":
            plt.plot(history['train_rmse'], label='Train RMSE')
            plt.plot(history['val_rmse'], label='Val RMSE')
            plt.xlabel('Epoch')
        elif model_name == "XGBoost":
            # XGBoost evals_result structure: {'train': {'rmse': [...]}, 'eval': {'rmse': [...]}}
            epochs = len(history['train']['rmse'])
            plt.plot(range(epochs), history['train']['rmse'], label='Train RMSE')
            plt.plot(range(epochs), history['eval']['rmse'], label='Val RMSE')
            plt.xlabel('Boosting Round')
            
        plt.ylabel('RMSE')
        plt.title(f'{model_name} Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{config.PLOTS_DIR}/{model_name}_learning_curve.pdf", dpi=600, bbox_inches="tight")
        plt.close()

    def plot_actual_vs_predicted(self, df, model_names):
        """Scatter plots of Actual vs Predicted for multiple models."""
        log("Generating Actual vs. Predicted plots...")
        n_models = len(model_names)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
        fig.suptitle('Actual vs. Predicted Dose')
        axes = axes.flatten()

        for i, model in enumerate(model_names):
            pred_col = f'Dose_Pred_{model}'
            if pred_col not in df.columns: continue

            ax = axes[i]
            sns.scatterplot(data=df, x=config.TARGET_COL, y=pred_col, ax=ax, alpha=0.5, s=15)
            
            # Identity line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Ideal')
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{model} Predictions')
            ax.set_xlabel('Actual Dose')
            ax.set_ylabel('Predicted Dose')
            
            # Metrics on plot
            y_true = df[config.TARGET_COL].values
            y_pred = df[pred_col].values
            r2 = r2_score(y_true, y_pred)
            mpe = mean_absolute_percentage_error(y_true, y_pred) * 100
            
            ax.text(0.05, 0.9, f"R2 = {r2:.3f}\nMAPE = {mpe:.2f}%", 
                    transform=ax.transAxes, fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            ax.legend()
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f"{config.PLOTS_DIR}/actual_vs_predicted.pdf", dpi=600, bbox_inches="tight")
        plt.close()

    def plot_residuals(self, df, model_names):
        """Residual plots (Error vs Actual)."""
        log("Generating Residual plots...")
        fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 6), sharey=True)
        if len(model_names) == 1: axes = [axes]
        
        fig.suptitle('Residual Plots (Actual - Predicted)')

        for i, model in enumerate(model_names):
            error_col = f'Error_{model}'
            if error_col not in df.columns: continue
            
            ax = axes[i]
            sns.scatterplot(data=df, x=config.TARGET_COL, y=error_col, ax=ax, alpha=0.5, s=15)
            ax.axhline(0, color='r', linestyle='--')
            ax.set_title(f'{model} Residuals')
            ax.set_xlabel('Actual Dose')
            ax.set_xscale('log') # Log scale often helps with Dose data
            ax.set_ylabel('Error')

        plt.tight_layout()
        plt.savefig(f"{config.PLOTS_DIR}/residuals.pdf", dpi=600, bbox_inches="tight")
        plt.close()

    def plot_feature_importance(self, importance_dict, model_name):
        """Bar chart for feature importance."""
        log(f"Plotting feature importance for {model_name}...")
        
        df_imp = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis', hue='Feature', legend=False)
        plt.title(f'{model_name} Feature Importance')
        plt.tight_layout()
        
        save_path = f"{config.PLOTS_DIR}/feature_importance_{model_name}.pdf"
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()

    def plot_model_comparison_violins(self, df, target_col, models_dict, category_col):
        """
        Comparing Actual vs Predicted distributions side-by-side using Violin plots.
        models_dict: {'XGB': 'Dose_Pred_XGB', 'RF': 'Dose_Pred_RF'...}
        """
        # Prepare layout
        n_plots = 1 + len(models_dict) # Actual + N models
        cols = 2
        rows = (n_plots + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 8 * rows), sharey=True)
        axes = axes.flatten()
        fig.suptitle(f"Dose Distribution by {category_col}", fontsize=22)

        # 1. Plot Actual
        log_target = f"{target_col}_log"
        if log_target not in df.columns:
            df[log_target] = np.log10(df[target_col])
            
        sns.violinplot(ax=axes[0], x=category_col, y=log_target, data=df)
        axes[0].set_title("Actual Distribution")
        
        # 2. Plot Models
        for i, (name, pred_col) in enumerate(models_dict.items(), start=1):
            log_pred = f"{pred_col}_log"
            df[log_pred] = np.log10(df[pred_col])
            
            sns.violinplot(ax=axes[i], x=category_col, y=log_pred, data=df)
            axes[i].set_title(f"{name} Predictions")
            axes[i].tick_params(axis='x', rotation=30)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/violin_comparison_{category_col}.pdf", dpi=600, bbox_inches="tight")
        plt.close()

    def plot_tabnet_masks(self, tabnet_model, X_data, feature_names, num_samples=100):
        """
        Generates and plots the 5x2 grid of all 10 step-wise masks.
        """
        log(f"📈 Plotting all {tabnet_model.n_steps} step masks for first {num_samples} samples...")

        # 1. Select subset and generate masks
        X_subset = X_data[:num_samples]
        _, masks = tabnet_model.explain(X_subset) # We only need 'masks' here
        
        n_features = len(feature_names)
        
        # 2. Create the 5x2 grid
        fig, ax = plt.subplots(5, 2, figsize=(30, 20), sharex=True, sharey=True) 

        for i in range(tabnet_model.n_steps):
            row = i // 2  # Integer division to get row (0-4)
            col = i % 2   # Modulo to get column (0-1)
            
            current_ax = ax[row, col]
            
            # 3. Plot the transpose (.T)
            im = current_ax.imshow(
                masks[i].T,   
                cmap='viridis', 
                aspect='auto'
            )
            
            # 4. Set labels and ticks
            current_ax.set_title(f'Decision Step {i+1} Mask')
            current_ax.set_xlabel(f'Samples (First {num_samples})')
            current_ax.set_ylabel('Features')
            current_ax.set_yticks(np.arange(n_features))
            current_ax.set_yticklabels(feature_names)
            
        cbar_ax = fig.add_axes([0.15, 0.01, 0.8, 0.02])
        
        # 5. Add colorbar
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Mask Importance')

        fig.suptitle(f'TabNet Feature Importance Masks (First {num_samples} Samples)', fontsize=20, y=1.02)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        # 6. Save and show
        plt.savefig(f"{self.output_dir}/tabnet_all_step_masks.pdf", dpi=600, bbox_inches="tight")
        plt.close()

    def plot_rf_performance(self, rf_model, X_test, y_test, scaler):
        """Plots RMSE vs Number of Trees."""
        log("Plotting RF Performance Curve...")
        preds = [tree.predict(X_test) for tree in rf_model.estimators_]
        preds = np.array(preds)
        
        # Cumulative average of predictions
        cum_preds = np.cumsum(preds, axis=0) / np.arange(1, len(preds) + 1)[:, None]
        
        rmse_scores = []
        # Inverse transform y_test once
        y_true = 10**scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

        for i in range(len(preds)):
            # Inverse transform the cumulative prediction at this step
            y_p = 10**scaler.inverse_transform(cum_preds[i].reshape(-1, 1)).ravel()
            rmse_scores.append(np.sqrt(mean_squared_error(y_true, y_p)))

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(preds) + 1), rmse_scores)
        plt.xlabel('Number of Trees')
        plt.ylabel('RMSE')
        plt.title('Random Forest Performance vs Trees')
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/rf_performance.pdf", dpi=600, bbox_inches="tight")
        plt.close()

    def plot_error_by_category(self, df, category_col, model_names):
        """Plots error metric for each category value."""
        results = []
        for model in model_names:
            err_col = f'Error_{model}'
            if err_col not in df.columns: continue

            df[f'MAPE_{model}'] = (df[err_col].abs() / df[config.TARGET_COL].abs()) * 100

            mape_by_cat = df.groupby(category_col)[f'MAPE_{model}'].mean().reset_index(name='MAPE')
            mape_by_cat['Model'] = model
            results.append(mape_by_cat)

        df_mape = pd.concat(results)

        # Plotting
        plt.figure(figsize=(18, 8))
        sns.barplot(data=df_mape, x=category_col, y='MAPE', hue='Model')
        plt.title(f'Mean Absolute % Error (MAPE) by {category_col}', fontsize=32)
        plt.xlabel(category_col)
        plt.ylabel('MAPE (%)')
        plt.xticks(rotation=75, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/mape_by_{category_col}.pdf", dpi=600, bbox_inches="tight")
        plt.close()

class VisualizerEDA:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        # Apply the user's specific theme
        sns.set_theme(
            style="whitegrid",
            rc={
                'font.family': ['DejaVu Serif'], 'font.size': 20,
                'axes.titlesize': 20, 'axes.labelsize': 18,
                'xtick.labelsize': 15, 'ytick.labelsize': 15,
                'legend.fontsize': 15, 'axes.linewidth': 2
            }
        )
        os.makedirs(output_dir, exist_ok=True)

    def plot_correlation_heatmap(self, df, cols, target_col):
        """Plots correlation matrix of numerical features vs target."""
        corr = df[cols + [target_col]].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_heatmap.pdf", dpi=600, bbox_inches="tight")
        plt.close()

    def plot_distribution_bf_af(self, data_before, data_after):
        import scipy.stats as stats

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Before Transformation
        sns.histplot(data_before, kde=True, ax=ax1, color='blue', bins=50)
        ax1.set_title(f'Original Dose Distribution')

        ax1.text(0.3, 0.95, f"Skew: {stats.skew(data_before):.2f}",
                transform=ax1.transAxes,
                ha='center', va='top', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Create inset zoom
        axin1 = ax1.inset_axes([0.59, 0.59, 0.36, 0.36])
        sns.histplot(data_before, kde=True, ax=axin1, color='blue', bins=50)
        axin1.set_xlim(2e-8, 5e-8)
        axin1.set_ylim(0, 500)
        axin1.set(xlabel='', ylabel='')
        ax1.indicate_inset_zoom(axin1)
        
        # Plot 2: After Transformation
        sns.histplot(data_after, kde=True, ax=ax2, color='green', bins=50)
        ax2.set_title(f'Log Transfromed Dose Distribution')

        ax2.text(0.3, 0.95, f"Skew: {stats.skew(data_after):.2f}",
                transform=ax2.transAxes,
                ha='center', va='top', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/dose_vs_log_hisplot.pdf", dpi=600, bbox_inches="tight")
        #plt.show()
        plt.close()

    def plot_histograms(self, dfA, dfB, col):
        if dfA is None or dfB is None:
            raise RuntimeError(
                'Please ensure you have generated the interpolated dataset by running "interpolate_akima_finer.py"'
            )

        plt.figure(figsize=(8, 5))
        sns.histplot(dfA[col], color="blue", label="Original Data", kde=True, stat="density")
        sns.histplot(dfB[col], color="red", label="New Data", kde=True, stat="density", alpha=0.6)
        plt.title(f"Comparison for Distribution of {col}")
        plt.legend()
        plt.savefig(f"{self.output_dir}/two_dataset_comparison_hist.pdf", dpi=600, bbox_inches="tight")
        #plt.show()
        plt.close()

    def plot_grouped_boxplots(self, df, category_col, value_col_before, value_col_after):
        print(f"Plotting grouped boxplots by '{category_col}'...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Before Transformation
        sns.boxplot(data=df, x=category_col, y=value_col_before, ax=ax1)
        ax1.set_title(f'{value_col_before} by {category_col}')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: After Transformation
        sns.boxplot(data=df, x=category_col, y=value_col_after, ax=ax2)
        ax2.set_title(f'{value_col_after} by {category_col}')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/dose_distn_by_{category_col}_boxplots.pdf", dpi=600, bbox_inches="tight")
        plt.close()

    def plot_dose_boxen(self, df, raw_target_col):
        df[f"{raw_target_col}_log"] = np.log10(df[raw_target_col])
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        # Plot 1: Radionuclide
        sns.boxenplot(ax=axes[0], x="Radionuclide", y=f"{raw_target_col}_log", data=df, hue="Radionuclide",palette="viridis", legend=False)
        axes[0].set_ylabel("Dose values (Log10 Transformed)", fontsize=18)
        axes[0].set_xlabel("Radionuclide", fontsize=20)
        axes[0].set_title("Distribution of Actual Dose Across Radionuclides", fontsize=20)
        axes[0].tick_params(axis='x', rotation=45, labelsize=18)
        axes[0].tick_params(axis='y', labelsize=18)
        # Plot 2: Stability Category
        sns.boxenplot(ax=axes[1], x="Stability Category", y=f"{raw_target_col}_log", data=df, hue="Stability Category",palette="viridis", legend=False)
        # axes[1].set_ylabel("Dose values (Log10 Transformed)", fontsize=20) # Shared Y, so usually omitted
        axes[1].set_xlabel("Stability Category", fontsize=20)
        axes[1].set_title("Distribution of Actual Dose Across Stability Category", fontsize=20)
        axes[1].tick_params(axis='x', rotation=30, labelsize=20)

        plt.tight_layout()
        save_path = f"{self.output_dir}/CatWise_Dose_boxen_plots.pdf"
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()