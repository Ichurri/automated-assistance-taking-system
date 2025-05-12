import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from utils import load_from_h5, encode_labels, ensure_directory

class FaceRecognitionModel:
    """
    Neural network model for face recognition based on HOG features.
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize the face recognition model.
        
        Args:
            model_dir (str): Directory to save trained models
        """
        self.model_dir = model_dir
        self.model = None
        self.label_encoder = None
        
        # Ensure model directory exists
        ensure_directory(self.model_dir)
    
    def build_model(self, input_dim, num_classes):
        """
        Build neural network architecture.
        
        Args:
            input_dim (int): Dimension of input features
            num_classes (int): Number of classes (persons) to recognize
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # Input layer
            Dense(512, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, features_file, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the neural network on HOG features.
        
        Args:
            features_file (str): Path to H5 file containing features and labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            dict: Training history
        """
        # Load features and labels
        features, labels = load_from_h5(features_file)
        
        if features is None or labels is None:
            return None
            
        # Encode string labels to integers
        encoded_labels, self.label_encoder = encode_labels(labels)
        
        # Convert to one-hot encoding
        label_binarizer = LabelBinarizer()
        one_hot_labels = label_binarizer.fit_transform(encoded_labels)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            features, one_hot_labels, test_size=validation_split, stratify=one_hot_labels
        )
        
        logging.info(f"Training with {X_train.shape[0]} samples, validating with {X_val.shape[0]} samples")
        logging.info(f"Input feature dimension: {features.shape[1]}")
        logging.info(f"Number of classes: {one_hot_labels.shape[1]}")
        
        # Build model
        self.model = self.build_model(features.shape[1], one_hot_labels.shape[1])
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True            ),
            ModelCheckpoint(
                os.path.join(self.model_dir, 'face_recognition_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model and label encoder
        self.save_model()
        
        # Plot training history
        self._plot_training_history(history)
        
        return history.history
    def save_model(self, model_filename='face_recognition_model.keras', 
                  classes_filename='label_encoder.npy'):
        """
        Save the trained model and label encoder.
        
        Args:
            model_filename (str): Filename for model
            classes_filename (str): Filename for label encoder
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if self.model is None:
            logging.error("No model to save")
            return False
            
        try:
            # Save model
            model_path = os.path.join(self.model_dir, model_filename)
            self.model.save(model_path)
            
            # Save label encoder
            if self.label_encoder is not None:
                encoder_path = os.path.join(self.model_dir, classes_filename)
                np.save(encoder_path, self.label_encoder.classes_)
                
            logging.info(f"Model and label encoder saved to {self.model_dir}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            return False
    def load_model(self, model_filename='face_recognition_model.keras', 
                  classes_filename='label_encoder.npy'):
        """
        Load a trained model and label encoder.
        
        Args:
            model_filename (str): Filename for model
            classes_filename (str): Filename for label encoder
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Load model
            model_path = os.path.join(self.model_dir, model_filename)
            self.model = load_model(model_path)
            
            # Load label encoder
            encoder_path = os.path.join(self.model_dir, classes_filename)
            self.label_encoder = LabelBinarizer()
            self.label_encoder.classes_ = np.load(encoder_path, allow_pickle=True)
            
            logging.info(f"Model and label encoder loaded from {self.model_dir}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def predict(self, hog_features):
        """
        Predict the identity of a face from its HOG features.
        
        Args:
            hog_features: HOG features extracted from a face image
            
        Returns:
            tuple: (predicted_label, confidence)
        """
        if self.model is None:
            logging.error("No model loaded")
            return None, 0
            
        # Reshape features if needed
        if len(hog_features.shape) == 1:
            hog_features = np.expand_dims(hog_features, axis=0)
            
        # Get prediction
        predictions = self.model.predict(hog_features)
        
        # Get class index and confidence
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        # Get label from class index
        if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
            predicted_label = self.label_encoder.classes_[class_idx]
        else:
            predicted_label = f"Person_{class_idx}"
            
        return predicted_label, confidence
    
    def _plot_training_history(self, history):
        """
        Plot and save training history.
        
        Args:
            history: Keras training history object
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 5))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Model Loss')
            plt.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.model_dir, 'training_history.png')
            plt.savefig(plot_path)
            logging.info(f"Training history plot saved to {plot_path}")
            
        except Exception as e:
            logging.error(f"Error plotting training history: {e}")
