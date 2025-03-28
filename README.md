# Live Facial Expression Detector

This project is a real-time facial expression detection system built with Python, OpenCV, and TensorFlow. It uses a  deep learning model is made using CNN to classify facial expressions from a webcam feed into three emotions: **Happy**, **Sad**, and **Angry** . The system detects faces, processes them, and overlays the predicted emotion with a confidence score on the video feed.

## Features
- Real-time face detection using Haar Cascade Classifier.
- Emotion classification with a pre-trained TensorFlow model (`expression_detector.h5`).
- Displays the processed face and live video feed with emotion labels.
- Debug output with raw prediction probabilities.

## Prerequisites
- **Python 3.6+**: Ensure Python is installed on your system.
- **Webcam**: A working webcam connected to your computer.
- **Dependencies**: Install the required Python libraries.
- **Model File**: The pre-trained model `expression_detector.h5` (assumed to be trained on the FER2013 dataset).

### Installation
1. **Clone or Download**: Get this script into a local directory.
2. **Install Dependencies**: Open a terminal and run:
   ```bash
   pip install opencv-python numpy tensorflow
