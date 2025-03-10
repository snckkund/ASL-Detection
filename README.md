# ASL Vision - American Sign Language Recognition

ASL Vision is a real-time American Sign Language recognition system that uses computer vision and deep learning to interpret hand gestures. The application can recognize ASL signs through your webcam or from uploaded images.

## Features

- Real-time ASL sign recognition using webcam
- Support for image upload and recognition
- Interactive web interface built with Streamlit
- Dataset visualization and statistics
- Model training capabilities
- Reference chart for ASL signs

## Requirements

- Python 3.12+
- Required packages listed in `requirements.txt`:
  - mediapipe >= 0.10.21
  - numpy >= 1.26.4
  - opencv-python >= 4.11.0.86
  - streamlit >= 1.42.1
  - tensorflow >= 2.18.0
  - tqdm >= 4.66.1

## Installation

1. Clone the repository:
```bash
git clone https://github.com/snckkund/ASL-Detection.git
cd ASL-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

You can download the ASL Alphabet dataset from Kaggle and place it in the `dataset` folder:
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## Project Structure

```
ASL-Detection
│
├── .github/                 # GitHub workflows
├── assets/                  # Directory for assets
├── backend/                 # Backend application files
│   ├── models/              # Directory for model files
│   └── main.py              # Main backend application file
├── dataset/                 # Training and testing dataset directory
├── frontend/                # Frontend application files
│   ├── app.js               # Main frontend application file
│   ├── index.html           # Main frontend HTML file
│   └── styles.css           # Main frontend CSS file
├── requirements.txt         # List of required packages
└── training/                # Directory for training related files
```

## Usage

### Camera Mode
- Click "Camera Mode" and press "Start Camera"
- Show hand signs in front of the camera
- Real-time predictions will be displayed

### Upload Mode
- Click "Upload Mode"
- Upload an image containing a hand sign
- View prediction results

## Model Details

- Uses MediaPipe for hand landmark detection
- TensorFlow model trained on hand landmark coordinates
- Supports 29 classes (A-Z + [del, space, nothing] in working)

## TODOs

### High Priority
- [ ] Fix prediction through upload image mode
  - Improve hand detection in static images
  - Adjust confidence thresholds for better detection
  - Add preprocessing steps for uploaded images

- [ ] Add special handling for specific classes
  - [ ] Implement "nothing" class detection
  - [ ] Improve "space" gesture recognition
  - [ ] Enhance "del" gesture detection

### Future Improvements
- [ ] Add continuous text prediction mode
- [ ] Implement word suggestions
- [ ] Add gesture history
- [ ] Improve prediction confidence scoring
- [ ] Add model retraining option

## Known Issues

1. Upload mode sometimes fails to detect hands in images
2. Special characters (del, space) need better detection accuracy
3. "Nothing" class needs special handling
