# PlumeDoseNet

*Interpolation-Driven Machine Learning Approaches for Plume Shine Dose Estimation: A Comparison of XGBoost, Random Forest, and TabNet.*

---

## Introduction

**[Project Description]** 
PlumeDoseNet is a machine learning–based framework for rapid plume shine dose estimation. The project employs XGBoost, Random Forest, and TabNet models trained on interpolation-enhanced datasets to provide fast, accurate, and physics-consistent surrogate predictions for radiological consequence assessment.

<img width="605" height="265" alt="psd" src="https://github.com/user-attachments/assets/811291da-5950-4f6d-948b-4231bc3e8d4b" />

---

## 📂 Project Structure

The repository is organized into a modular structure to separate data processing, model logic, and execution scripts.

```text
PlumeDoseNet/
├── Library/                     # Raw and processed datasets
│   ├── filtered_distance...csv  # Input training data
│   └── ...
├── src/                         # Core library package
│   ├── data_manager.py          # Data loading, encoding, and splitting
│   ├── models.py                # XGBoost, RF, and TabNet wrapper classes
│   ├── visualization.py         # Plotting utilities (EDA, Metrics, Masks)
│   ├── evaluation.py            # Metric calculations (RMSE, MAPE, sMAPE)
│   └── utils.py                 # Helper functions and logging
├── interpolate_akima_finer.py   # Step 1: Generates interpolated dataset
├── Create_interp_train_test.py  # Step 2: Creates strict train/test splits
├── config.py                    # Central configuration (Paths, Hyperparams)
├── main_eda.py                  # Exploratory Data Analysis script
├── main_optuna.py               # Hyperparameter optimization for TabNet
├── main_train.py                # Main training execution script
└── main_evaluate.py             # Inference and evaluation script
└── app_pdosenet.py              # Streamlit based GUI; command to run: streamlit run app_pdosenet.py


```

---

## Getting Started

### Prerequisites

Make sure you have Conda installed. Then create and activate the project environment & install dependences:

```bash
conda create -n plumedose python=3.10 -y
conda activate plumedose
pip install -r requirements.txt
```

---

## Execution Workflow

Follow these steps in order to reproduce the results.

### 1. Data Generation & Preprocessing

Before training, we must generate the interpolated dataset and create a strictly separated test set to prevent data leakage.

* **Step 1: Interpolation** Generates a finer grid of data points using Akima interpolation. The geenrated dataset is also available at Zenodo for download. Save it in "Library" directory.
```bash
python interpolate_akima_finer.py

```


* **Step 2: Train/Test Split** Splits the interpolated data into training and testing sets, ensuring test points are unseen in the training phase.
```bash
python Create_interp_train_test.py

```



### 2. Configuration

Check `config.py` to ensure file paths and global hyperparameters (like `NUM_EPOCHS` or `SEED`) match your environment.

### 3. Exploratory Data Analysis (EDA)

Visualize the dataset distributions, correlations, and physical consistency before training.

```bash
python main_eda.py

```

*Outputs are saved to `plots/eda/`.*

### 4. Hyperparameter Tuning (Optional)

Run Optuna to find the best hyperparameters for the TabNet model. This will create a local SQLite database and save the best model configuration. Once you get the optimized hyper parameter space, it can be used directly in TabNet wrapper for next command.

```bash
python main_optuna.py

```

*Best parameters are logged and the optimized model is saved automatically.*

### 5. Training Models

Train the models (XGBoost, Random Forest, TabNet). You will be prompted to select which models to run. 

```bash
python main_train.py

```

* **Interactive Mode:** Type `xgboost`, `tabnet`, or `all` when prompted.
* **Outputs:** Trained models are saved to `output_dir/`.

N.B: Trained models and associated files are avaialble for download from Zenodo. save it in "saved_model" folder inside parent directory for directly using saved models without retraining.

### 6. Evaluation

Evaluate the trained models on the independent test set generated in Step 1. This script generates performance metrics (RMSE, MAPE, sMAPE) and comparative plots.

```bash
python main_evaluate.py

```

*Outputs:*

* **Metrics:** Saved to `output_dir/saved_metrics/`
* **Plots:** Saved to `plots/` (Learning curves, Feature importance, Residuals, Violin plots).

---


### 7. Web-based GUI app

A web-based graphical user interface (GUI) has been developed to enable interactive plume shine dose prediction using the trained machine learning models. The application allows users to provide scenario-specific inputs, including radionuclide, downwind distance, release height, and atmospheric stability category. Based on the selected inputs, the GUI generates plume shine dose predictions using the trained XGBoost, Random Forest, and TabNet models, along with the corresponding reference (true) dose values for comparison.

The interface is implemented using Streamlit and is intended for rapid scenario evaluation, model intercomparison, and demonstration purposes.

> [!IMPORTANT]
> Before running the GUI app, make sure the file paths inside app_pdosenet.py correctly point to your models, encoders, scaler files and supporting artifacts.

- **Option 1 — Use Pretrained Models (Quick Start)** \
  Download all required files from Zenodo: 10.5281/zenodo.18266001 \
  Create a folder named: ` saved_models/ ` \
  Place all **.json** and **.pkl** files inside it. Ensure paths in app_pdosenet.py point to ` saved_models/. ` \
  Run:
  ```bash
  streamlit run app_pdosenet.py
  ```

- **Option 2 — Use Self-Trained Models** \
  Train models using:
  ```bash
  python main_train.py
  ```
  
  Models will be saved in: ` output_dir/ ` \
  Update file paths inside load_artifacts() in app_pdosenet.py to point to output_dir/ instead of saved_models/.

### 🤝 Contributing

Contributions are welcome.
If you would like to improve the codebase, add features, or report issues, please feel free to fork the repository and submit a pull request. Constructive suggestions and discussions are also encouraged.

### 📄 License

This project is released under the MIT License.
See the LICENSE file for full license details.







