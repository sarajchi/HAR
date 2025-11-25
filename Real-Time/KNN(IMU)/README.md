# KNN(IMU) Real-Time
This folder contains the codebase and outputs associated with the Real-Time KNN (IMU) model.

# Start

**1_KNN_IMU_PrerecordedData.py**

Runs KNN predictions using the pre-recorded IMU dataset stored in Temporary_Data/.

This script is fully offline and does not require any physical IMUs.
It loads the trained KNN model, extracts features from each CSV, and reports the predicted action.

**2_GetData.py**

Captures live IMU data from two x-IMU3 sensors.
It streams accelerometer/gyroscope values, formats them into time windows, and saves each window into Temporary_Data/ for real-time processing.

**3_KNN_IMU_Real-Time.py**

Performs real-time action recognition using the trained KNN model.
The script monitors the Temporary_Data/ folder, loads each newly created IMU window from 2_GetData.py, extracts features, and outputs the classification result.

This script must run simultaneously with 2_GetData.py, because it depends on the live data windows that 2_GetData.py produces.
