
# Sign Language Recognition System Using Deep Learning and OpenCV

## Project Overview
- Objective: Detect and classify ASL hand signs (A-Z and 0-9) from webcam feed.
- Workflow:
  - Detect hand in real-time.
  - Collect cropped, standardized images for each sign.
  - Train a classifier (using Google Teachable Machine).
  - Run real-time sign classification.

## Features
- Real-time hand detection and tracking
- Automatic data collection with consistent 300x300 white background images
- Easy extension to more signs/alphabet classes
- Model training using Google Teachable Machine (simple UI, fast experimentation)
- Live sign recognition

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- CVZone (`cvzone`)
- Mediapipe (`mediapipe`)
- Numpy
- TensorFlow/Keras
- Webcam

## Installation
pip install opencv-python cvzone mediapipe numpy tensorflow
If you experience issues or compatibility errors, use package versions (OpenCV 6.0, TensorFlow 2.9.1).

## Data Collection Instructions
1. Run the data collection script (`data_collection.py` or similar).
2. Organize data using folders: `data/a`, `data/b`, `data/c`.
3. Use your webcam to position your hand in the required sign pose.
4. Press the `s` key to save images—collect approximately 300 images per class for better results.
5. Ensure varied backgrounds, angles, and lighting for robustness.
6. Images are auto-cropped and centered to 300x300 with white background.

## Model Training Instructions
1. Go to [Google Teachable Machine](https://teachablemachine.withgoogle.com/).
2. Start an *Image Project* and create classes (A-Z and 0-9).
3. Upload the respective images for each class.
4. Train the model with default parameters.
5. Download the TensorFlow/Keras model (`.h5` file) and labels (`labels.txt`).
6. Save these files in the `model/` directory of your project.

## Model Testing Instructions
1. Run the testing script (`test.py`).
2. The script will load the trained model and labels.
3. Real-time prediction from your webcam will identify the hand sign and display the classification result above the detected hand.
4. Results show the class (A-Z and 0-9) and confidence scores.

## Folder Structure
project-root/
├── data/
│   ├── a/
│   ├── b/
│   └── c/
├── model/
│   ├── keras_model.h5
│   └── labels.txt
├── data_collection.py
├── test.py
└── README.md

## Useful Links
- [CVZone Documentation](https://www.computervision.zone/)
- [Google Teachable Machine](https://teachablemachine.withgoogle.com/)
