# KNN(IMU) Offline
This folder contains the KNN(IMU) codes and outcomes. 


# KNN IMU Feature Extraction
The following image shows the IMU feature extraction technique.

<img width="657" height="367" alt="Untitled" src="https://github.com/user-attachments/assets/303b1bb4-e21a-47a6-aadb-533d036213da" />

# Dataset
The dataset for Human Action Recognition with IMUs and a Camera is provided at the following link:
https://drive.google.com/file/d/1tn91HX9y28Xy6x3cmq3V3b6SHgiwPRpx/view?usp=sharing

# Sliding Window
A sliding window of length 10 is used for Video and IMU data. They move forward in both the Video stream and the IMU data stream to ensure that the data fed into the model is synchronised and maintains the same length.

![Slide1](https://github.com/user-attachments/assets/693f3f88-dbf2-4c0a-a154-583661ee114c)

# Model
A new multimodal human action recognition model is proposed, which uses MoVinet to process the video data and LSTM to extract features from the IMU data, and finally fuses the above two features to classify the three actions of grasping, walking and placing during the process of carrying a box.
![Fig4 (1)](https://github.com/user-attachments/assets/4c561323-f942-4d44-ad9c-bb4513139d40)

# Supplementary Material:
A video demonstrating the study is available at the following link: https://youtu.be/PU3ySZ0spoI.

# Start
  
  RUN
  "1. Feature Extraction_IMU.py": This Python code is optional and presents the "All_IMU_Features.csv", which is already provided in the repository.
  "2. KNN_IMU_Results.py": This Python code presents the outcomes of the KNN(IMU) model, including the Confusion Matrix, 
  pip install -r requirements.tx
  
  python train_eval.py
