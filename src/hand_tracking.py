import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def process_frame(self, frame):
        """Process a frame and return hand landmarks."""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks with custom style
                landmark_spec = self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=4)
                connection_spec = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                
                # Draw landmarks and connections
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_spec,
                    connection_spec
                )
                
                # Get dimensions for bounding box
                h, w, _ = frame.shape
                landmarks = results.multi_hand_landmarks[0].landmark
                coords = [(int(l.x * w), int(l.y * h)) for l in landmarks]
                x_min = min(x for x, y in coords)
                x_max = max(x for x, y in coords)
                y_min = min(y for x, y in coords)
                y_max = max(y for x, y in coords)
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)
                
                return frame, (x_min, y_min, x_max, y_max)
        
        return frame, None
    
    def get_hand_crop(self, frame, bbox):
        """Crop the hand region from the frame."""
        if bbox is None:
            return None
        x_min, y_min, x_max, y_max = bbox
        return frame[y_min:y_max, x_min:x_max]

    def get_prediction(self, model, landmarks):
        """Get prediction only if confidence exceeds threshold."""
        prediction = model.predict(np.expand_dims(landmarks, 0), verbose=0)
        confidence = np.max(prediction[0])
        
        if confidence >= self.confidence_threshold:
            return IDX_TO_CLASS[np.argmax(prediction[0])], confidence
        return "uncertain", confidence
