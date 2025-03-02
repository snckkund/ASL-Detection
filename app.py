import os
import cv2
import time
import base64
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
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

# Detect environment
IS_HUGGINGFACE = "SPACE_ID" in os.environ
IS_LOCAL = not IS_HUGGINGFACE

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hand_tracker = HandTracker(confidence_threshold=0.7)

# Define paths
CHART_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'chart.jpg')

@st.cache_resource
def load_cached_model():
    """Load and cache the model to prevent reloading"""
    if os.path.exists(MODEL_PATH):
        with st.spinner('Loading ASL Detection model... This may take a few moments.'):
            if IS_HUGGINGFACE:
                # Optimize for Hugging Face environment
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                tf.config.set_visible_devices([], 'GPU')
            # Load model with reduced memory usage
            model = load_trained_model(MODEL_PATH)
            return model
    return None

def init_session_state():
    if 'mode' not in st.session_state:
        st.session_state.mode = 'camera'  # Default to camera for all environments
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'model' not in st.session_state:
        st.session_state.model = load_cached_model()
    if 'hand_tracker' not in st.session_state:
        st.session_state.hand_tracker = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    if 'camera_permission_requested' not in st.session_state:
        st.session_state.camera_permission_requested = False

def check_camera_permission():
    """Check if camera access is allowed"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception as e:
        print(f"Camera access error: {str(e)}")
        return False

def request_camera_access():
    """Request camera access from browser"""
    if st.session_state.get('camera_permission_requested'):
        return
        
    st.session_state.camera_permission_requested = True
    st.components.v1.html("""
        <div id="camera_permission">Requesting camera access...</div>
        <script>
            async function requestCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: {
                            width: { ideal: 640 },
                            height: { ideal: 480 },
                            frameRate: { ideal: 30 }
                        }
                    });
                    document.getElementById('camera_permission').innerText = 'Camera access granted!';
                    stream.getTracks().forEach(track => track.stop());
                } catch (err) {
                    document.getElementById('camera_permission').innerText = 'Camera access denied: ' + err.message;
                    console.error('Error:', err);
                }
            }
            requestCamera();
        </script>
    """, height=50)
    time.sleep(1)  # Brief delay for permission dialog

def initialize_camera():
    """Initialize camera with proper settings"""
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        else:
            st.error("Could not open camera. Please check permissions.")
            return None
    except Exception as e:
        st.error(f"Error initializing camera: {str(e)}")
        return None

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
                pass
            st.session_state.hands = None
            
        st.session_state.camera_active = False
    except Exception as e:
        st.error(f"Error during cleanup: {str(e)}")

def show_reference_chart():
    """Display ASL reference chart"""
    try:
        # Load and display the reference chart image
        if os.path.exists(CHART_PATH):
            image = Image.open(CHART_PATH)
            # Resize image to be more compact while maintaining aspect ratio
            width, height = image.size
            new_width = 600  # Set a smaller fixed width
            new_height = int((new_width * height) / width)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a container with custom styling
            with st.container():
                st.markdown("""
                    <style>
                    .reference-chart {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 5px;  # Reduced padding
                        background-color: white;
                        margin-bottom: 10px;  # Reduced margin
                        box-shadow: none;  /* Remove white bar effect */
                    }
                    </style>
                    <div class="reference-chart">
                    """, unsafe_allow_html=True)
                st.image(image, use_container_width=True, caption="ASL Reference Chart")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Reference chart not found. Please check the assets directory.")
    except Exception as e:
        st.error(f"Error loading reference chart: {str(e)}")

def stop_camera():
    """Stop and release camera resources"""
    if 'cap' in st.session_state:
        st.session_state.cap.release()
        st.session_state['camera_active'] = False
        st.success('Camera stopped.')

def run_camera_feed():
    """Run the camera feed continuously"""
    if IS_HUGGINGFACE:
        # For Hugging Face, use browser-based camera
        st.components.v1.html("""
            <div style="position: relative;">
                <video id="webcam" autoplay playsinline style="width: 100%; max-width: 640px;"></video>
                <canvas id="canvas" style="display: none;"></canvas>
            </div>
            <script>
                const video = document.getElementById('webcam');
                const canvas = document.getElementById('canvas');
                const ctx = canvas.getContext('2d');

                async function setupCamera() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({
                            video: {
                                width: 640,
                                height: 480,
                                frameRate: 30
                            }
                        });
                        video.srcObject = stream;
                        await video.play();
                        
                        // Set canvas size to match video
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        
                        // Start sending frames
                        sendFrames();
                        
                    } catch (error) {
                        console.error('Error:', error);
                    }
                }

                function sendFrames() {
                    if (video.readyState === video.HAVE_ENOUGH_DATA) {
                        // Draw video frame to canvas
                        ctx.drawImage(video, 0, 0);
                        // Get frame data
                        const imageData = canvas.toDataURL('image/jpeg', 0.8);
                        // Send to Python
                        window.parent.postMessage({
                            type: 'video_frame',
                            data: imageData
                        }, '*');
                    }
                    // Continue sending frames
                    requestAnimationFrame(sendFrames);
                }

                setupCamera();
            </script>
        """, height=500)
        
        # Add frame receiver
        st.components.v1.html("""
            <script>
                window.addEventListener('message', function(event) {
                    if (event.data.type === 'video_frame') {
                        window.streamlit.setComponentValue({
                            type: 'frame',
                            data: event.data.data
                        });
                    }
                });
            </script>
        """, height=0)
        
        # Process frames if available
        if 'component_value' in st.session_state:
            try:
                frame_data = st.session_state.component_value.get('data')
                if frame_data:
                    # Convert base64 image to numpy array
                    encoded_data = frame_data.split(',')[1]
                    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Convert to RGB for MediaPipe
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Process with MediaPipe
                        results = st.session_state.hand_tracker.process(frame_rgb)
                        
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                # Draw landmarks
                                mp_drawing.draw_landmarks(
                                    frame_rgb,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style()
                                )
                                
                                # Get predictions
                                landmarks = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark]).flatten()
                                prediction = st.session_state.model.predict(np.expand_dims(landmarks, 0), verbose=0)
                                predicted_class = IDX_TO_CLASS[np.argmax(prediction[0])]
                                confidence = np.max(prediction[0])
                                
                                # Draw prediction text
                                h, w, _ = frame_rgb.shape
                                x_min = int(min(l.x * w for l in hand_landmarks.landmark))
                                y_min = int(min(l.y * h for l in hand_landmarks.landmark))
                                
                                text = f"{predicted_class} ({confidence:.1%})"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 1
                                thickness = 2
                                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                                
                                # Draw background rectangle
                                cv2.rectangle(frame_rgb, 
                                            (x_min - 10, y_min - text_size[1] - 20),
                                            (x_min + text_size[0] + 10, y_min),
                                            (0, 0, 0), -1)
                                
                                # Draw text
                                cv2.putText(frame_rgb, text,
                                          (x_min, y_min - 10), font,
                                          font_scale, (255, 255, 255), thickness)
                        
                        # Display processed frame
                        st.image(frame_rgb, channels="RGB", use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error processing frame: {str(e)}")
    else:
        # For local environment, use OpenCV camera
        FRAME_WINDOW = st.empty()
        try:
            while True:
                if not hasattr(st.session_state, 'cap') or st.session_state.cap is None:
                    break

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
                            x_min = int(min(l.x * w for l in hand_landmarks.landmark))
                            y_min = int(min(l.y * h for l in hand_landmarks.landmark))
                            
                            text = f"{predicted_class} ({confidence:.1%})"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1
                            thickness = 2
                            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                            
                            cv2.rectangle(frame_rgb, 
                                        (x_min - 10, y_min - text_size[1] - 20),
                                        (x_min + text_size[0] + 10, y_min),
                                        (0, 0, 0), -1)
                            
                            cv2.putText(frame_rgb, text,
                                      (x_min, y_min - 10), font,
                                      font_scale, (255, 255, 255), thickness)

                    FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error processing frame: {str(e)}")
                    cleanup()
                    break

                time.sleep(0.033)
                
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
            cleanup()

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

def main():
    st.set_page_config(page_title="ASL Detection System", layout="wide")
    init_session_state()
    
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "Camera"  # Default to camera for all environments

    # Sidebar
    st.sidebar.title("Navigation")
    
    # Show appropriate pages based on environment
    if IS_LOCAL:
        available_pages = ["Dataset Info", "Train Model", "Test Model"]
    else:
        available_pages = ["Test Model"]
        st.sidebar.info("⚡ Running on Hugging Face")
    
    page = st.sidebar.radio("Go to", available_pages)

    if page == "Dataset Info" and IS_LOCAL:
        dataset_info_page()
    elif page == "Train Model" and IS_LOCAL:
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
    
    if st.session_state.model is None:
        st.error("Model not found. Please train the model first.")
        return
    
    # Add reference chart in a smaller expander
    with st.expander("📖 ASL Reference Chart", expanded=False):
        show_reference_chart()

    # Mode selection with toggle buttons
    col1, col2 = st.columns(2)
    
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

    st.markdown("---")

    # Camera Mode
    if st.session_state.get('mode') == 'camera':
        # Create two columns for camera controls
        control_col1, control_col2 = st.columns(2)
        
        with control_col1:
            if not st.session_state.get('camera_active', False):
                if st.button("📷 Start Camera", key="start_camera"):
                    request_camera_access()
                    
                    # Initialize camera
                    if 'cap' in st.session_state and st.session_state.cap is not None:
                        st.session_state.cap.release()
                    
                    st.session_state.cap = initialize_camera()
                    if st.session_state.cap is not None:
                        st.session_state['camera_active'] = True
                        st.success("Camera initialized successfully!")
                        st.experimental_rerun()  # Force rerun to start the camera feed

        with control_col2:
            if st.session_state.get('camera_active', False):
                if st.button("⏹️ Stop Camera", key="stop_camera"):
                    cleanup()
                    st.experimental_rerun()

        # Run camera feed if active
        if st.session_state.get('camera_active', False):
            run_camera_feed()

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
                            st.image(image_rgb, caption="Original Image", use_column_width=True)
                        with col2:
                            st.image(image_with_landmarks, caption="Detected Hand Landmarks", use_column_width=True)
                        with col3:
                            st.markdown(f"""
                                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                                    <h2 style='color: #00acee;'>Prediction Results</h2>
                                    <p style='font-size: 24px;'>Sign: {predicted_class}</p>
                                    <p style='font-size: 18px;'>Confidence: {confidence:.2%}</p>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.image(image_rgb, caption="Original Image", use_column_width=True)
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