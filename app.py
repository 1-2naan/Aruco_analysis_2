import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def calculate_metrics(data, peaks):
    speeds = [0]  # Starting with an initial average speed of 0
    velocities = [0]  # Starting with 0 velocity
    accelerations = [0]  # Starting with 0 acceleration
    
    for i in range(1, len(peaks)):
        distance_diff = data['Distance'].iloc[peaks[i]] - data['Distance'].iloc[peaks[i-1]]
        time_diff = data['Timestamp'].iloc[peaks[i]] - data['Timestamp'].iloc[peaks[i-1]]
        average_speed = distance_diff / time_diff if time_diff else 0
        speeds.append(average_speed)
        
        velocity = distance_diff / time_diff if time_diff else 0
        velocities.append(velocity)
        
        acceleration = (velocities[i] - velocities[i-1]) / time_diff if i > 1 and time_diff else 0
        accelerations.append(acceleration)
        
    return speeds, velocities, accelerations

def plot_hand_movement(data, max_width, min_width, max_threshold_multiplier, min_threshold_multiplier):
    y = data['Distance'].values
    timestamps = data['Timestamp'].values
    median_val = np.median(y)
    
    max_threshold = median_val * max_threshold_multiplier
    min_threshold = median_val * min_threshold_multiplier
    
    max_peaks, _ = find_peaks(-y, width=max_width)
    min_peaks, _ = find_peaks(y, width=min_width)
    
    filtered_max_peaks = max_peaks[y[max_peaks] < max_threshold]
    filtered_min_peaks = min_peaks[y[min_peaks] > min_threshold]
    
    all_peaks = np.sort(np.concatenate((filtered_max_peaks, filtered_min_peaks)))
    
    speeds, velocities, accelerations = calculate_metrics(data, all_peaks)
    
    peaks_df = pd.DataFrame({
        'Timestamp': data['Timestamp'][all_peaks],
        'Distance': data['Distance'][all_peaks],
        'Average Speed': speeds,
        'Velocity': velocities,
        'Acceleration': accelerations
    })

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, y, label='Hand Movement')
    plt.axhline(y=max_threshold, color='g', linestyle='--', label='Max Peak Threshold')
    plt.axhline(y=min_threshold, color='y', linestyle='--', label='Min Peak Threshold')
    plt.plot(timestamps[filtered_max_peaks], y[filtered_max_peaks], "bo", label='Maximum Peaks')
    plt.plot(timestamps[filtered_min_peaks], y[filtered_min_peaks], "ro", label='Minimum Peaks')
    plt.xlabel('Timestamp')
    plt.ylabel('Distance Moved')
    plt.title('Hand Movement with Dynamic Thresholds')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)
    
    return peaks_df

# Streamlit app starts here
st.title('Hand Movement Visualization App')

uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type="csv")

for uploaded_file in uploaded_files:
    st.write(f"Processing file: {uploaded_file.name}")

    df = pd.read_csv(uploaded_file)
    df=df.iloc[:, :5]
    df.columns=['X','Y','Z','T','Frame_num']
    df['Timestamp'] = df['Frame_num'] / 60  # Assuming 60 FPS for simplicity
    df['X1'] = df['X'] - df['X'][0]
    df['Y1'] = df['Y'] - df['Y'][0]
    df['Z1'] = df['Z'] - df['Z'][0]
    df['Distance'] = np.sqrt(df['X1']**2 + df['Y1']**2 + df['Z1']**2)
    df2 = df[['Timestamp', 'Distance']]

    st.write("Adjust the peak width and threshold settings for this file:")
    max_width = st.slider(f'Select Maximum Peak Width for {uploaded_file.name}', 1, 200, 30, key=f"max_width_{uploaded_file.name}")
    min_width = st.slider(f'Select Minimum Peak Width for {uploaded_file.name}', 1, 200, 40, key=f"min_width_{uploaded_file.name}")
    max_threshold_multiplier = st.slider('Maximum Peak Threshold Multiplier', 0.5, 2.0, 1.0, 0.1, key=f"max_threshold_{uploaded_file.name}")
    min_threshold_multiplier = st.slider('Minimum Peak Threshold Multiplier', 1.0, 3.0, 1.5, 0.1, key=f"min_threshold_{uploaded_file.name}")

    peaks_df = plot_hand_movement(df2, max_width, min_width, max_threshold_multiplier, min_threshold_multiplier)

    st.download_button(
        "Download Peaks Data as CSV",
        data=peaks_df.to_csv(index=False).encode('utf-8'),
        file_name=f'{uploaded_file.name.split(".")[0]}_peaks_data.csv',
        mime='text/csv',
    )
