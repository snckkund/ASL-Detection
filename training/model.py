import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import NUM_CLASSES
import numpy as np

# Enable GPU memory growth to prevent memory allocation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured for memory growth")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Using CPU for training")

def create_landmark_model():
    """Create model that uses hand landmarks for classification."""
    model = models.Sequential([
        # Input shape: (21 landmarks * 3 coordinates (x,y,z))
        layers.Input(shape=(63,)),
        
        # Dense layers for landmark processing
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Use label smoothing to reduce overconfidence
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    return model

def train_landmark_model(model, train_data, train_labels, validation_data=None, epochs=20):
    """Train the landmark-based model with validation."""
    history = model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        validation_data=validation_data,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_landmark_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
    )
    return history

def load_trained_model(model_path):
    """Load a trained model from disk."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        return None

class ASLEnsemble:
    def __init__(self, num_models=3):
        self.models = [create_landmark_model() for _ in range(num_models)]
    
    def train(self, train_data, train_labels, validation_data=None, epochs=20):
        """Train multiple models for ensemble prediction."""
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}")
            train_landmark_model(
                model,
                train_data,
                train_labels,
                validation_data=validation_data,
                epochs=epochs
            )
    
    def predict(self, landmarks):
        """Get ensemble prediction from all models."""
        predictions = [model.predict(np.expand_dims(landmarks, 0), verbose=0) 
                      for model in self.models]
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def save_models(self, base_path='ensemble_models'):
        """Save all models in the ensemble."""
        for i, model in enumerate(self.models):
            model.save(f'{base_path}_model_{i}.h5')
    
    def load_models(self, base_path='ensemble_models'):
        """Load all models in the ensemble."""
        self.models = []
        i = 0
        while True:
            try:
                model = tf.keras.models.load_model(f'{base_path}_model_{i}.h5')
                self.models.append(model)
                i += 1
            except:
                break
