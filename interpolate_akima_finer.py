import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# Load dataset
file_path = "Library/filtered_distance_2000_height_200_train_99.csv"
df = pd.read_csv(file_path)

# Ensure Dose is numeric and drop NaNs
df['Dose'] = pd.to_numeric(df['Dose'], errors='coerce')
df = df.dropna(subset=['Dose'])

# Get unique values
release_heights = sorted(df["Release Height (m)"].unique())
radionuclides = [
    "Cs-137", "Cs-134", "Ar-41", "Xe-135", "Co-60", "I-131", "I-132",
    "Kr-87", "Kr-88", "Kr-85", "Sr-85", "Ru-103", "Ru-106", "Na-22",
    "Eu-152", "Eu-154", "Eu-155"
]

# Function to perform finer PCHIP interpolation
def interpolate_pchip_finer(df, radionuclide, release_height):
    interpolated_data = []
    
    # Filter data for the given Radionuclide and Release Height
    filtered_df = df[(df["Release Height (m)"] == release_height) & (df["Radionuclide"] == radionuclide)]
    
    for stability_category in range(1, 7):
        stability_filtered_df = filtered_df[filtered_df["Stability Category"] == stability_category].drop_duplicates()
        stability_filtered_df = stability_filtered_df.groupby('Distance (m)', as_index=False)['Dose'].mean()

        # Extract values
        distance = stability_filtered_df['Distance (m)'].values
        dose = stability_filtered_df['Dose'].values

        # Perform interpolation if sufficient points exist
        if len(distance) > 1:
            pchip_interp = PchipInterpolator(distance, dose)
            fine_distances = np.linspace(distance.min(), distance.max(), 2000)  # Increased to 2000 points
            interpolated_dose = pchip_interp(fine_distances)

            # Append interpolated results
            for d, dose_interp in zip(fine_distances, interpolated_dose):
                interpolated_data.append([radionuclide, release_height, stability_category, d, dose_interp])

    # Convert to DataFrame
    interpolated_df = pd.DataFrame(interpolated_data, columns=[
        "Radionuclide", "Release Height (m)", "Stability Category", "Distance (m)", "Dose"
    ])
    
    return interpolated_df

# Generate finer interpolated data for all radionuclides and release heights
all_finer_interpolated_data = []

for release_height in release_heights:
    for radionuclide in radionuclides:
        interpolated_df = interpolate_pchip_finer(df, radionuclide, release_height)
        all_finer_interpolated_data.append(interpolated_df)

# Merge all interpolated data into a single DataFrame
final_finer_interpolated_df = pd.concat(all_finer_interpolated_data, ignore_index=True)

# Save the finer interpolated dataset
output_finer_path = "Library/finer_interpolated_data_filtered_distance_2000_height_200_train_99.csv"
final_finer_interpolated_df.to_csv(output_finer_path, index=False)

# Return the output file path
#output_finer_path

