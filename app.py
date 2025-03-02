import streamlit as st
import cv2
import numpy as np
import os
import time
import base64
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
            let videoElement = null;
            
            async function requestCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: {
                            width: { ideal: 640 },
                            height: { ideal: 480 },
                            frameRate: { ideal: 30 }
                        }
                    });
                    
                    // Create video element if it doesn't exist
                    if (!videoElement) {
                        videoElement = document.createElement('video');
                        videoElement.style.display = 'none';
                        document.body.appendChild(videoElement);
                    }
                    
                    // Connect stream to video element
                    videoElement.srcObject = stream;
                    await videoElement.play();
                    
                    document.getElementById('camera_permission').innerText = 'Camera access granted!';
                    window.streamActive = true;
                } catch (err) {
                    document.getElementById('camera_permission').innerText = 'Camera access denied: ' + err.message;
                    console.error('Error:', err);
                    window.streamActive = false;
                }
            }
            requestCamera();
        </script>
    """, height=50)
    time.sleep(1)  # Brief delay for permission dialog

def initialize_browser_camera():
    """Initialize camera in browser environment"""
    try:
        st.components.v1.html("""
            <div style="display: none;">
                <video id="camera" autoplay playsinline></video>
                <canvas id="canvas"></canvas>
            </div>
            <script>
                const video = document.getElementById('camera');
                const canvas = document.getElementById('canvas');
                const context = canvas.getContext('2d');
                
                async function startCamera() {
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
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        
                        // Store stream in window object for later access
                        window.cameraStream = stream;
                        
                        // Start frame capture loop
                        function captureFrame() {
                            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                                const imageData = canvas.toDataURL('image/jpeg');
                                window.parent.postMessage({type: 'video_frame', data: imageData}, '*');
                            }
                            requestAnimationFrame(captureFrame);
                        }
                        captureFrame();
                        
                        return true;
                    } catch (err) {
                        console.error('Error:', err);
                        document.body.innerHTML += '<div style="color: red;">Camera error: ' + err.message + '</div>';
                    }
                }
                
                // Start camera when component loads
                startCamera().then(success => {
                    window.cameraInitialized = success;
                });
            </script>
        """)
        # Wait briefly for camera initialization
        time.sleep(2)
        return True
    except Exception as e:
        st.error(f"Camera initialization error: {str(e)}")
        return False

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
        # For Hugging Face environment, use Python MediaPipe with Streamlit camera
        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # Configure MediaPipe with more sensitive detection
        hands = mp_hands.Hands(
            static_image_mode=True,  # Set to True for better accuracy
            max_num_hands=1,
            min_detection_confidence=0.3,  # Lower threshold for better detection
            min_tracking_confidence=0.3
        )

        # Create a container for the video feed
        video_container = st.container()
        
        # Use Streamlit's camera input
        camera = st.camera_input("Camera Feed")
        
        if camera is not None:
            try:
                # Convert the image from bytes to numpy array
                img_array = np.frombuffer(camera.getvalue(), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # To improve detection, process the image
                rgb_frame = cv2.flip(rgb_frame, 1)  # Mirror the image
                
                # Process frame with MediaPipe
                results = hands.process(rgb_frame)
                
                # Add debug info
                debug_text = "No hands detected"
                
                # Draw hand landmarks with custom settings
                if results.multi_hand_landmarks:
                    debug_text = f"Detected {len(results.multi_hand_landmarks)} hand(s)"
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw skeleton
                        mp_drawing.draw_landmarks(
                            rgb_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(
                                color=(0, 255, 0),  # Green color
                                thickness=2,
                                circle_radius=2
                            ),
                            connection_drawing_spec=mp_drawing.DrawingSpec(
                                color=(255, 255, 255),  # White color
                                thickness=2
                            )
                        )
                        
                        # Draw colored dots for each landmark
                        for id, landmark in enumerate(hand_landmarks.landmark):
                            height, width, _ = rgb_frame.shape
                            cx, cy = int(landmark.x * width), int(landmark.y * height)
                            cv2.circle(rgb_frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # Blue dots
                            
                        # Get hand landmarks for prediction
                        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                        
                        # Make prediction if model is loaded
                        if st.session_state.model:
                            # Prepare landmarks for prediction
                            landmarks_flat = landmarks.flatten()
                            prediction = st.session_state.model.predict(landmarks_flat.reshape(1, -1))
                            predicted_class = IDX_TO_CLASS[np.argmax(prediction)]
                            confidence = np.max(prediction)
                            
                            debug_text += f" - Predicted: {predicted_class} ({confidence:.2f})"
                            
                            # Draw prediction text
                            cv2.putText(
                                rgb_frame,
                                f"{predicted_class} ({confidence:.2f})",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),  # Green text
                                2
                            )
                
                # Draw debug text
                cv2.putText(
                    rgb_frame,
                    debug_text,
                    (10, rgb_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # White text
                    1
                )
                
                # Display the processed frame
                with video_container:
                    st.image(rgb_frame, channels="RGB")  # Note: Using RGB since we're already in RGB colorspace
            
            except Exception as e:
                st.error(f"Error processing frame: {str(e)}")
                st.error(f"Error details: {type(e).__name__}")
                import traceback
                st.error(traceback.format_exc())
        
        # Clean up
        hands.close()
    else:
        FRAME_WINDOW = st.empty()
        while True:
            try:
                if not hasattr(st.session_state, 'cap') or st.session_state.cap is None:
                    break

                ret, frame = st.session_state.cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with hand tracker
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
                        
                        # Get landmarks and make prediction
                        landmarks = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark]).flatten()
                        prediction = st.session_state.model.predict(np.expand_dims(landmarks, 0), verbose=0)
                        predicted_class = IDX_TO_CLASS[np.argmax(prediction[0])]
                        confidence = np.max(prediction[0])

                        # Draw prediction on frame
                        h, w, _ = frame_rgb.shape
                        coords = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]
                        x_min = max(0, min(x for x, y in coords) - 20)
                        y_min = max(0, min(y for x, y in coords) - 20)
                        cv2.putText(frame_rgb, f"{predicted_class} ({confidence:.4f})",
                                  (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.9, (0, 255, 0), 2)

                # Display the frame
                FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)
                time.sleep(0.033)  # Cap at ~30 FPS
                
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
                break

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
        # Camera controls
        control_col1, control_col2 = st.columns(2)
        
        with control_col1:
            if not st.session_state.get('camera_active', False):
                if st.button('Start Camera'):
                    if IS_HUGGINGFACE:
                        st.session_state['camera_active'] = True
                        st.success("Camera initialized successfully!")
                    else:
                        if 'cap' in st.session_state and st.session_state.cap is not None:
                            st.session_state.cap.release()
                        
                        st.session_state.cap = cv2.VideoCapture(1)
                        if st.session_state.cap.isOpened():
                            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            st.session_state.cap.set(cv2.CAP_PROP_FPS, 30)
                            st.session_state['camera_active'] = True
                            st.success("Camera initialized successfully!")
                        else:
                            st.error("Could not initialize camera. Please check permissions and try again.")

        with control_col2:
            if st.session_state.get('camera_active', False):
                if st.button("⏹️ Stop Camera", key="stop_camera"):
                    cleanup()
                    st.rerun()

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