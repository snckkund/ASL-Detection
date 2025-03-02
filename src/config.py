import os

# Model Configuration
IMG_SIZE = 224  # Still needed for initial image processing
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 29

# Number of landmarks and their dimensions
NUM_LANDMARKS = 21  # MediaPipe hand landmarks
LANDMARK_DIMS = 3   # x, y, z coordinates

# Dataset paths
TRAIN_DIR = "dataset/asl_alphabet_train/asl_alphabet_train/"
TEST_DIR = "dataset/asl_alphabet_test/asl_alphabet_test/"

# Model paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "asl_landmark_model.h5")
ENSEMBLE_MODEL_BASE_PATH = os.path.join(MODEL_DIR, "ensemble_model")

# Training parameters
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5

# Confidence threshold for predictions
PREDICTION_THRESHOLD = 0.7

# Data augmentation parameters
ROTATION_RANGE = 15  # degrees
SCALE_RANGE = 0.1   # proportion

# Class mapping
CLASS_NAMES = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26,
    'nothing': 27, 'space': 28
}

# Reverse mapping
IDX_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}

# MediaPipe hand landmark configuration
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}
