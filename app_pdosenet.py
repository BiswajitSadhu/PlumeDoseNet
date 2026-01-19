import streamlit as st
import numpy as np
import xgboost as xgb
import joblib
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# Streamlit Page Config
# =========================================================
st.set_page_config(
    page_title="PlumeDoseNet Dashboard",
    page_icon="☁️",
    layout="wide" 
)

st.title("Plume Shine Dose Comparison")
st.markdown("Compare AI model predictions against Ground Truth (PlumeShine).")

# =========================================================
# 1. Load Resources (Cached)
# =========================================================
@st.cache_resource
def load_artifacts():   
    # --- Load Models ---
    xgb_model = xgb.Booster()
    xgb_model.load_model("saved_models/xgboost_model_final_ep100_dep30.json") ##
    #xgb_model.load_model("saved_models/xgboost_trained_model_14022025.json") ##
    
    rf_model = joblib.load("saved_models/random_forest_final_ep100_dep15.pkl") ##
    #rf_model = joblib.load("saved_models/random_forest_model_new.pkl") ##

    tabnet_model = joblib.load("saved_models/tabnet_final.pkl")

    # --- Load Saved Preprocessors ---
    with open("saved_models/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    with open("saved_models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Extract class names for dropdowns
    radionuclides = label_encoders["Radionuclide"].classes_.tolist()
    stability_categories = label_encoders["Stability Category"].classes_.tolist()

    return xgb_model, rf_model, tabnet_model, label_encoders, scaler, radionuclides, stability_categories

try:
    xgb_model, rf_model, tabnet_model, label_encoders, scaler, radionuclides, stability_categories = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Please ensure the 'saved_models' folder exists and contains the required .pkl/.json files.")
    st.stop()

# =========================================================
# 2. Logic Helpers
# =========================================================
def prepare_input_vector(rad, stab, rh, dist):
    # Transform categories using the loaded encoders
    rad_enc = label_encoders["Radionuclide"].transform([rad])[0]
    stab_enc = label_encoders["Stability Category"].transform([stab])[0]
    
    # Ensure this order matches exactly what was used in training!
    return np.array([[rad_enc, stab_enc, rh, dist]])

def inverse_transform_log_dose(norm_pred, scaler):
    # Inverse MinMax
    log_dose = scaler.inverse_transform([[norm_pred]])[0, 0]
    # Inverse Log10
    actual_dose = 10 ** log_dose
    return actual_dose

# =========================================================
# 3. Sidebar UI
# =========================================================
with st.sidebar:
    st.header("⚙️ Input Parameters")
    
    # Categorical Inputs
    selected_rad = st.selectbox("Radionuclide", radionuclides)
    selected_stab = st.selectbox("Stability Category", stability_categories)
    
    # Numerical Inputs
    st.markdown("---")
    release_height = st.number_input("Release Height (m)", min_value=0.0, value=10.0, step=1.0)
    distance = st.number_input("Distance (m)", min_value=1.0, value=2000.0, step=10.0)

    # Model Selection
    st.markdown("---")
    st.subheader("Model Selection")
    model_options = ["XGBoost", "Random Forest", "TabNet", "PLUMNESHINE"]
    selected_models = st.multiselect("Choose models to run:", model_options, default=["XGBoost", "Random Forest", "PLUMNESHINE"])
    
    run_btn = st.button("Run Prediction")

# =========================================================
# 4. Main Execution Logic
# =========================================================
if run_btn:
    if not selected_models:
        st.warning("Please select at least one model to run.")
    else:
        # Prepare input vector once
        input_vec = prepare_input_vector(selected_rad, selected_stab, release_height, distance)
        
        results_dict = {}     # Stores formatted strings for display
        numeric_dict = {}     # Stores raw floats for plotting
        
        # --- XGBoost ---
        if "XGBoost" in selected_models:
            dmat = xgb.DMatrix(input_vec)
            pred_norm_xgb = xgb_model.predict(dmat)[0]
            # XGB was trained on Log10 + Scaled data
            dose_xgb = inverse_transform_log_dose(pred_norm_xgb, scaler) ##

            # If using model: saved_models/xgboost_trained_model_14022025.json
            #dose_xgb = pred_norm_xgb ##
            
            results_dict["XGBoost"] = dose_xgb
            numeric_dict["XGBoost"] = dose_xgb

        # --- Random Forest ---
        if "Random Forest" in selected_models:
            pred_norm_rf = rf_model.predict(input_vec)[0]
            # RF Trained on Log10 + Scaled data
            dose_rf = inverse_transform_log_dose(pred_norm_rf, scaler) ##

            # If using model: saved_models/random_forest_model_new.pkl
            #dose_rf = pred_norm_rf ##
            
            results_dict["Random Forest"] = dose_rf
            numeric_dict["Random Forest"] = dose_rf

        # --- TabNet ---
        if "TabNet" in selected_models:
            # TabNet requires float32
            pred_array = tabnet_model.predict(input_vec.astype(np.float32))[0]
            pred_norm_tab = pred_array.item()
            # TabNet was trained on Log10 + Scaled data
            dose_tab = inverse_transform_log_dose(pred_norm_tab, scaler)
            
            results_dict["TabNet"] = dose_tab
            numeric_dict["TabNet"] = dose_tab

        # --- PLUMNESHINE (Ground Truth) ---
        if "PLUMNESHINE" in selected_models:
            try:
                from ps_ground_truth import PLUMNESHINE
                
                # Map stability categories to fit PLUMNESHINE config requirements
                stability_map = {
                    1: '1', 2: '2', 3: '3', 
                    4: '4', 5: '5', 6: '6',
                    'A': '1', 'B': '2', 'C': '3', 
                    'D': '4', 'E': '5', 'F': '6'
                }
                stab_config_val = stability_map.get(selected_stab, str(selected_stab))

                config_ps = {
                    "release_height": release_height,
                    "measurement_height": 1.0,
                    "rads_list": [selected_rad],
                    "instantaneous_release_bq_list": [1],
                    "spatial_distance": distance,
                    "stability_category": stab_config_val,
                }
                
                # Suppress output if necessary or handle delays
                with st.spinner("Calculating Ground Truth ....."):
                    ps_model = PLUMNESHINE(device="cpu", config=config_ps)
                    dose_ps = ps_model.plumeshine_dose()
                
                results_dict["PLUMNESHINE"] = dose_ps
                numeric_dict["PLUMNESHINE"] = dose_ps
                
            except ImportError:
                st.error("Module 'ps_ground_truth' not found. Ensure the file exists in the directory.")
            except Exception as e:
                st.error(f"PLUMNESHINE Error: {e}")

        # =========================================================
        # 5. Display Results
        # =========================================================
        
        # Create columns for layout: [Results Table, Plot]
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📝 Predictions")
            for model_name, dose_val in results_dict.items():
                # Formatting: Highlight Ground Truth
                if model_name == "PLUMNESHINE":
                    st.success(f"**{model_name}:**\n\n `{dose_val:.6e}` μSv")
                else:
                    st.info(f"**{model_name}:**\n\n `{dose_val:.6e}` μSv")

        with col2:
            st.subheader("📊 Comparison Chart")
            if len(numeric_dict) > 0:
                # Prepare data for chart
                chart_data = pd.DataFrame({
                    "Model": list(numeric_dict.keys()),
                    "Dose (μSv/hr)": list(numeric_dict.values())
                }).set_index("Model")
                
                # Streamlit's native bar chart for interactivity
                st.bar_chart(chart_data, color="#4A90E2")
                
                # Optional: Matplotlib fallback for static images with exact formatting
                # fig, ax = plt.subplots(figsize=(8, 4))
                # sns.barplot(x=list(numeric_dict.keys()), y=list(numeric_dict.values()), ax=ax)
                # st.pyplot(fig)
