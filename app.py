import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import platform
import mediapipe as mp

from src.model import create_landmark_model, load_trained_model, ASLEnsemble
from src.data_utils import (
    create_dataset, process_uploaded_image, get_dataset_stats,
    plot_dataset_distribution, get_sample_images, extract_hand_landmarks, create_landmark_dataset
)
from src.config import (
    TRAIN_DIR, TEST_DIR, MODEL_PATH, MODEL_DIR,
    IDX_TO_CLASS, IMG_SIZE, EPOCHS
)
from src.hand_tracking import HandTracker

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe components
hand_tracker = HandTracker(confidence_threshold=0.7)

def init_session_state():
    if 'mode' not in st.session_state:
        st.session_state.mode = 'camera'
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'model' not in st.session_state:
        if os.path.exists(MODEL_PATH):
            st.session_state.model = load_trained_model(MODEL_PATH)
    if 'hand_tracker' not in st.session_state:
        st.session_state.hand_tracker = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

def initialize_camera():
    """Initialize camera with better error handling."""
    camera = None
    
    # Try multiple camera indices and backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_ANY, "Default"),
        (None, "Legacy")
    ]
    
    for index in range(3):  # Try first 3 camera indices
        for backend, backend_name in backends:
            try:
                if backend is not None:
                    camera = cv2.VideoCapture(index, backend)
                else:
                    camera = cv2.VideoCapture(index)
                
                if camera and camera.isOpened():
                    ret, _ = camera.read()
                    if ret:
                        print(f"Camera initialized successfully on index {index} with {backend_name}")
                        # Set camera properties
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        camera.set(cv2.CAP_PROP_FPS, 30)
                        return camera
                    camera.release()
            except Exception as e:
                print(f"Failed to initialize camera {index} with {backend_name}: {str(e)}")
                if camera:
                    camera.release()
                continue
    
    # If we get here, no camera was successfully initialized
    st.error("""
    Camera initialization failed. Please try:
    1. Refreshing the page
    2. Checking camera permissions in your browser
    3. Disconnecting and reconnecting your camera
    4. Closing other applications that might be using the camera
    5. Using a different USB port
    """)
    return None

def get_available_cameras():
    """Test available camera indices and return working ones."""
    available_cameras = {}
    
    # Try indices 0 to 10
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Add camera to available list
                    available_cameras[i] = f"Camera {i}"
                cap.release()
        except:
            continue
    
    return available_cameras

def cleanup():
    """Clean up camera resources safely"""
    try:
        if hasattr(st.session_state, 'cap') and st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
            
        if hasattr(st.session_state, 'hands') and st.session_state.hands is not None:
            try:
                st.session_state.hands.close()
            except:
                pass  # Ignore MediaPipe close errors
            st.session_state.hands = None
            
        st.session_state.camera_active = False
    except Exception as e:
        st.error(f"Error during cleanup: {str(e)}")

def get_sample_image(sign):
    # Get a sample image from the training dataset
    train_dir = "dataset/asl_alphabet_train/asl_alphabet_train"
    sign_dir = os.path.join(train_dir, sign)
    if os.path.exists(sign_dir):
        # Get first image from the directory
        images = [f for f in os.listdir(sign_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            return os.path.join(sign_dir, images[0])
    return None

def show_reference_chart():
    # Create columns for the reference chart
    cols = st.columns(6)  # 6 columns for compact display
    
    # Group the signs into categories
    alphabets = [chr(i) for i in range(65, 91)]  # A to Z
    special = ['del', 'nothing', 'space']
    all_signs = alphabets + special
    
    # Display signs in a grid layout
    for idx, sign in enumerate(all_signs):
        col_idx = idx % 6
        with cols[col_idx]:
            st.write(f"**{sign}**")
            # Get and display sample image
            img_path = get_sample_image(sign)
            if img_path:
                img = Image.open(img_path)
                # Resize image to make it smaller and consistent
                img = img.resize((100, 100))
                st.image(img, use_container_width=True)

def main():
    st.set_page_config(page_title="ASL Detection System", layout="wide")
    init_session_state()
    
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "Camera"

    if 'camera_index' not in st.session_state:
        st.session_state.camera_index = 0

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dataset Info", "Train Model", "Test Model"])

    if page == "Dataset Info":
        dataset_info_page()
    elif page == "Train Model":
        train_page()
    else:
        test_page()

def dataset_info_page():
    st.header("Dataset Information")

    # Display dataset statistics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Dataset")
        if os.path.exists(TRAIN_DIR):
            train_stats, train_total = get_dataset_stats(TRAIN_DIR)
            st.write(f"Total training images: {train_total}")

            # Plot distribution
            st.subheader("Training Data Distribution")
            fig = plot_dataset_distribution(train_stats)
            if fig:
                st.pyplot(fig)
                plt.close(fig)  # Clean up
        else:
            st.warning("Training directory not found!")

    with col2:
        st.subheader("Testing Dataset")
        if os.path.exists(TEST_DIR):
            test_stats, test_total = get_dataset_stats(TEST_DIR)
            st.write(f"Total testing images: {test_total}")

            # Plot distribution
            st.subheader("Testing Data Distribution")
            fig = plot_dataset_distribution(test_stats)
            if fig:
                st.pyplot(fig)
                plt.close(fig)  # Clean up
        else:
            st.warning("Testing directory not found!")

    # Display sample images
    st.subheader("Sample Images from Training Set")
    if os.path.exists(TRAIN_DIR):
        samples = get_sample_images(TRAIN_DIR)
        cols = st.columns(5)
        for idx, (class_name, image) in enumerate(list(samples.items())[:5]):
            with cols[idx]:
                st.image(image, caption=class_name)

def train_page():
    st.header("Model Training")

    if not os.path.exists(TRAIN_DIR):
        st.error("Training directory not found! Please make sure the dataset is properly set up.")
        return

    train_stats, train_total = get_dataset_stats(TRAIN_DIR)
    if train_total == 0:
        st.error("No training images found! Please check the dataset directory structure.")
        return

    st.info(f"Training on {train_total} images across {len(train_stats)} classes")

    if st.button("Train New Model"):
        # Create model
        model = create_landmark_model()

        # Create landmark datasets
        landmark_progress = st.progress(0)
        landmark_status = st.empty()
        train_landmarks = []
        train_labels = []
        
        valid_data_count = 0
        with st.spinner("Extracting hand landmarks from training data..."):
            total_classes = len(os.listdir(TRAIN_DIR))
            for idx, class_name in enumerate(os.listdir(TRAIN_DIR)):
                class_path = os.path.join(TRAIN_DIR, class_name)
                if not os.path.isdir(class_path):
                    continue
                
                # Update progress
                progress = (idx + 1) / total_classes
                landmark_progress.progress(progress)
                landmark_status.text(f"Processing class: {class_name} ({idx + 1}/{total_classes})")
                
                # Process images in this class
                class_landmarks, class_labels = create_landmark_dataset(class_path, hand_tracker)
                if class_landmarks is not None and class_labels is not None:
                    train_landmarks.extend(class_landmarks)
                    train_labels.extend(class_labels)
                    valid_data_count += len(class_landmarks)
                    landmark_status.text(f"Processing class: {class_name} ({idx + 1}/{total_classes}) - Found {len(class_landmarks)} valid samples")

        if valid_data_count == 0:
            st.error("Failed to create training dataset! No valid hand landmarks were detected.")
            return

        st.info(f"Successfully extracted {valid_data_count} hand landmarks across {len(train_stats)} classes")

        train_landmarks = np.array(train_landmarks)
        train_labels = np.array(train_labels)
        st.success("Landmark extraction completed!")

        # Training progress tracking
        st.markdown("### Training Progress")
        epochs_progress = st.progress(0)
        epoch_status = st.empty()
        metrics_placeholder = st.empty()
        
        # Plot containers
        loss_chart = st.empty()
        accuracy_chart = st.empty()
        
        # Initialize metrics tracking
        losses = []
        accuracies = []

        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                # Update progress bar
                progress = (epoch + 1) / EPOCHS
                epochs_progress.progress(progress)
                epoch_status.text(f"Epoch {epoch + 1}/{EPOCHS}")
                
                # Update metrics
                if logs:
                    loss = logs.get('loss', 0)
                    accuracy = logs.get('accuracy', 0)
                    losses.append(loss)
                    accuracies.append(accuracy)
                    
                    # Display current metrics
                    metrics_placeholder.markdown(f"""
                        **Current Metrics:**
                        - Loss: {loss:.4f}
                        - Accuracy: {accuracy:.4f}
                    """)
                    
                    # Plot training progress
                    fig_loss, ax_loss = plt.subplots()
                    ax_loss.plot(losses)
                    ax_loss.set_title('Training Loss')
                    ax_loss.set_xlabel('Epoch')
                    ax_loss.set_ylabel('Loss')
                    loss_chart.pyplot(fig_loss)
                    plt.close(fig_loss)
                    
                    fig_acc, ax_acc = plt.subplots()
                    ax_acc.plot(accuracies)
                    ax_acc.set_title('Training Accuracy')
                    ax_acc.set_xlabel('Epoch')
                    ax_acc.set_ylabel('Accuracy')
                    accuracy_chart.pyplot(fig_acc)
                    plt.close(fig_acc)

        # Train model
        with st.spinner("Training model..."):
            try:
                model.fit(
                    train_landmarks, train_labels,
                    epochs=EPOCHS,
                    callbacks=[ProgressCallback()]
                )

                # Save model
                os.makedirs(MODEL_DIR, exist_ok=True)
                model.save(MODEL_PATH)
                st.session_state.model = model
                st.success("Model training completed!")
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

def test_page():
    st.title("Real-time ASL Detection")
    
    if 'model' not in st.session_state:
        if os.path.exists(MODEL_PATH):
            st.session_state.model = load_trained_model(MODEL_PATH)
        else:
            st.error("Model not found. Please train the model first.")
            return

    # Add reference chart in an expander
    with st.expander("📖 ASL Reference Chart", expanded=True):
        show_reference_chart()
    
    # Mode selection with toggle buttons
    col1, col2 = st.columns(2)
    
    # Style the buttons to look like toggle switches
    def get_button_style(active):
        return f"""
        <style>
        div.stButton > button {{
            background-color: {'#00acee' if active else '#ffffff'};
            color: {'white' if active else 'black'};
            width: 100%;
            padding: 10px;
            border: {'none' if active else '1px solid #ddd'};
            border-radius: 5px;
        }}
        </style>
        """
    
    with col1:
        st.markdown(get_button_style(st.session_state.get('mode') == 'camera'), unsafe_allow_html=True)
        if st.button("📷 Camera Mode", key="camera_mode"):
            st.session_state.mode = 'camera'
            cleanup()
            st.rerun()
    
    with col2:
        st.markdown(get_button_style(st.session_state.get('mode') == 'upload'), unsafe_allow_html=True)
        if st.button("📤 Upload Mode", key="upload_mode"):
            st.session_state.mode = 'upload'
            cleanup()
            st.rerun()

    st.markdown("---")  # Add separator

    # Camera Mode
    if st.session_state.get('mode') == 'camera':
        # Camera controls
        control_col1, control_col2 = st.columns(2)
        frame_placeholder = st.empty()
        
        with control_col1:
            if not st.session_state.get('camera_active', False):
                if st.button("▶️ Start Camera", key="start_camera"):
                    st.session_state.cap = cv2.VideoCapture(0)
                    if not st.session_state.cap.isOpened():
                        st.error("Could not open camera. Please check permissions.")
                        return
                    st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    st.session_state.cap.set(cv2.CAP_PROP_FPS, 30)
                    st.session_state.camera_active = True
        
        with control_col2:
            if st.session_state.get('camera_active', False):
                if st.button("⏹️ Stop Camera", key="stop_camera"):
                    cleanup()
                    st.rerun()

        # Main camera loop
        if st.session_state.get('camera_active', False) and hasattr(st.session_state, 'cap'):
            try:
                while st.session_state.camera_active:
                    ret, frame = st.session_state.cap.read()
                    if not ret:
                        st.error("Failed to read from camera")
                        cleanup()
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    try:
                        results = st.session_state.hand_tracker.process(frame_rgb)
                        
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    frame_rgb,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style()
                                )

                                landmarks = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark]).flatten()
                                prediction = st.session_state.model.predict(np.expand_dims(landmarks, 0), verbose=0)
                                predicted_class = IDX_TO_CLASS[np.argmax(prediction[0])]
                                confidence = np.max(prediction[0])

                                h, w, _ = frame_rgb.shape
                                coords = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]
                                x_min = max(0, min(x for x, y in coords) - 20)
                                y_min = max(0, min(y for x, y in coords) - 20)
                                cv2.putText(frame_rgb, f"{predicted_class} ({confidence:.2%})",
                                          (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                          0.9, (0, 255, 0), 2)

                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                    except Exception as e:
                        print(f"Frame processing error: {e}")
                        continue

                    time.sleep(0.03)

            except Exception as e:
                st.error(f"Camera error: {str(e)}")
                cleanup()
            finally:
                cleanup()

    # Upload Mode
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                # Load and process image
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                
                # Create a hand tracker specifically for static images
                image_hand_tracker = mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.2,  # Lower threshold for images
                    model_complexity=1
                )
                
                # Process with MediaPipe
                results = image_hand_tracker.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    # Create a copy for drawing
                    image_with_landmarks = image_rgb.copy()
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        mp_drawing.draw_landmarks(
                            image_with_landmarks,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                        # Extract landmarks and predict
                        landmarks = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark]).flatten()
                        prediction = st.session_state.model.predict(np.expand_dims(landmarks, 0), verbose=0)
                        predicted_class = IDX_TO_CLASS[np.argmax(prediction[0])]
                        confidence = np.max(prediction[0])

                        # Show both original and processed images
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(image_rgb, caption="Original Image", use_container_width=True)
                        with col2:
                            st.image(image_with_landmarks, caption="Detected Hand Landmarks", use_container_width=True)
                        with col3:
                            st.markdown(f"""
                                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                                    <h2 style='color: #00acee;'>Prediction Results</h2>
                                    <p style='font-size: 24px;'>Sign: {predicted_class}</p>
                                    <p style='font-size: 18px;'>Confidence: {confidence:.2%}</p>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.image(image_rgb, caption="Original Image", use_container_width=True)
                    st.error("No hand detected in the image. Please ensure the hand is clearly visible.")
                
                # Clean up the image hand tracker
                image_hand_tracker.close()
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.write("Error details:", str(e))

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()