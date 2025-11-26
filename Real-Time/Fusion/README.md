# Fusion(IMU+Vison) Real-Time
This folder contains the codebase and outputs associated with the Real-Time Fusion model.

# Start

**1_Fusion_PrerecordingData.py**

Runs Fusion predictions using the pre-recorded dataset stored in Temporary_Data/.

This script is offline and does not require any physical IMUs or a Camera.
It loads the trained Fusion model, extracts features from each CSV, and reports the predicted action.

**2_GetData.py**

Captures live IMU and Vision data from two x-IMU3 sensors and an egocentric camera.
It streams accelerometer/gyroscope values and camera frames, formats them into time windows, and saves each window into Temporary_Data/ for real-time processing.

**3_Fusion_Real-Time.py**

Performs real-time action recognition using the trained Fusion model.
The script monitors the Temporary_Data/ folder, loads each newly created IMU window from 2_GetData.py, extracts features, and outputs the classification result.

This script must run simultaneously with 2_GetData.py, because it depends on the live data windows that 2_GetData.py produces.
