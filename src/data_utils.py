import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from src.config import IMG_SIZE, BATCH_SIZE, CLASS_NAMES
import cv2

def preprocess_image(image):
    """Preprocess a single image for model input."""
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def process_uploaded_image(uploaded_file):
    """Process an uploaded image file."""
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    image = np.array(image)
    image = preprocess_image(image)
    return tf.expand_dims(image, 0)

def get_class_from_filename(filename):
    """Extract class name from filename, handling both training and test patterns."""
    # Remove extension
    base_name = os.path.splitext(filename)[0]
    # Handle test file pattern (e.g., "A_test" -> "A")
    if "_test" in base_name:
        class_name = base_name.split("_")[0].upper()
    else:
        class_name = base_name.upper()
    return class_name

def create_dataset(directory):
    """Create a TensorFlow dataset from directory."""
    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")

    try:
        # Check if directory has subdirectories (training data structure)
        has_subdirs = any(os.path.isdir(os.path.join(directory, d)) for d in os.listdir(directory))

        if has_subdirs:
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='categorical',
                image_size=(IMG_SIZE, IMG_SIZE),
                batch_size=BATCH_SIZE,
                shuffle=True
            )
        else:
            # Handle flat directory structure (test data)
            images = []
            labels = []
            for file in os.listdir(directory):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_name = get_class_from_filename(file)
                    if class_name in CLASS_NAMES:
                        img_path = os.path.join(directory, file)
                        img = tf.io.read_file(img_path)
                        img = tf.image.decode_image(img, channels=3)
                        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
                        images.append(img)
                        label = tf.one_hot(CLASS_NAMES[class_name], len(CLASS_NAMES))
                        labels.append(label)

            if not images:
                return None

            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            dataset = dataset.batch(BATCH_SIZE)

        return dataset.map(lambda x, y: (x/255.0, y))
    except Exception as e:
        print(f"Error loading dataset from {directory}: {str(e)}")
        return None

def get_dataset_stats(directory):
    """Get statistics about the dataset."""
    if not os.path.exists(directory):
        return {}, 0

    stats = {}
    total_images = 0

    # Check if directory has subdirectories
    has_subdirs = any(os.path.isdir(os.path.join(directory, d)) for d in os.listdir(directory))

    if has_subdirs:
        # Handle training data structure (subdirectories)
        subdirs = [d for d in os.listdir(directory) 
                  if os.path.isdir(os.path.join(directory, d))]

        for class_name in subdirs:
            class_path = os.path.join(directory, class_name)
            num_images = len([f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            stats[class_name] = num_images
            total_images += num_images
    else:
        # Handle test data structure (flat directory)
        for file in os.listdir(directory):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_name = get_class_from_filename(file)
                if class_name in CLASS_NAMES:
                    stats[class_name] = stats.get(class_name, 0) + 1
                    total_images += 1

    return stats, total_images

def plot_dataset_distribution(stats):
    """Create a bar plot of dataset distribution."""
    if not stats:
        return None

    # Set style to default white background
    plt.style.use('default')

    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(15, 6))

    classes = list(stats.keys())
    values = list(stats.values())

    # Create bars with a specific color
    bars = ax.bar(classes, values, color='#2196F3', alpha=0.7)

    # Customize plot
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax.set_title('Distribution of Images Across Classes', fontsize=14, pad=20)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig

def get_sample_images(directory, num_samples=1):
    """Get sample images from each class."""
    if not os.path.exists(directory):
        return {}

    samples = {}
    has_subdirs = any(os.path.isdir(os.path.join(directory, d)) for d in os.listdir(directory))

    if has_subdirs:
        # Handle training data structure
        subdirs = [d for d in os.listdir(directory) 
                  if os.path.isdir(os.path.join(directory, d))]
        for class_name in subdirs:
            class_path = os.path.join(directory, class_name)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                sample_path = os.path.join(class_path, images[0])
                samples[class_name] = Image.open(sample_path)
    else:
        # Handle test data structure
        for file in os.listdir(directory):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_name = get_class_from_filename(file)
                if class_name in CLASS_NAMES:
                    sample_path = os.path.join(directory, file)
                    samples[class_name] = Image.open(sample_path)

    return samples

def extract_hand_landmarks(image, hand_tracker):
    """Extract normalized hand landmarks from image."""
    results = hand_tracker.hands.process(image)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        # Flatten landmarks into 1D array of (x,y,z) coordinates
        coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
        return np.array(coords).flatten()
    return None

def create_landmark_dataset(directory, hand_tracker):
    """Create dataset of hand landmarks instead of images."""
    landmarks_data = []
    labels = []
    
    try:
        class_name = os.path.basename(directory).upper()
        if class_name not in CLASS_NAMES:
            print(f"Skipping invalid class directory: {class_name}")
            return None, None
            
        # Special handling for "nothing" class only
        if class_name == "NOTHING":
            num_samples = len([f for f in os.listdir(directory) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if num_samples > 0:
                # Create zero vectors for "nothing" class
                landmarks = np.zeros((num_samples, 63))
                labels = np.array([CLASS_NAMES[class_name]] * num_samples)
                return landmarks, tf.keras.utils.to_categorical(labels, len(CLASS_NAMES))
            return None, None
            
        # Normal processing for all other classes (including DEL and SPACE)
        for image_file in os.listdir(directory):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(directory, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hand_tracker.hands.process(image_rgb)
            if results.multi_hand_landmarks:
                # Get landmarks from the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark]).flatten()
                landmarks_data.append(landmarks)
                labels.append(CLASS_NAMES[class_name])
    
        if not landmarks_data:
            print(f"No valid landmarks found in directory: {directory}")
            return None, None
            
        return np.array(landmarks_data), tf.keras.utils.to_categorical(labels, len(CLASS_NAMES))
        
    except Exception as e:
        print(f"Error processing directory {directory}: {str(e)}")
        return None, None

def augment_landmarks(landmarks, rotation_range=15, scale_range=0.1):
    """Apply augmentation to landmark coordinates."""
    # Convert back to 2D array of (x,y,z) coordinates
    landmarks_2d = landmarks.reshape(-1, 3)
    
    # Random rotation
    angle = np.random.uniform(-rotation_range, rotation_range)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    # Random scale
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
    
    # Apply transformations
    landmarks_2d = landmarks_2d @ rotation_matrix * scale
    
    return landmarks_2d.flatten()