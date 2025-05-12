import os
import logging
import datetime
import h5py
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder

# Configure logging
def setup_logger(log_dir='logs'):
    """Set up and configure logger for the application."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'face_recognition_{timestamp}.log')
    
    # Configure logger
    logger = logging.getLogger('face_recognition')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create directories if they don't exist
def ensure_directory(directory):
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

# Save dataset to H5 file
def save_to_h5(features, labels, file_path):
    """Save features and labels to H5 file."""
    try:
        with h5py.File(file_path, 'w') as h5f:
            # Save features
            h5f.create_dataset('features', data=features)
            
            # Convert string labels to ASCII for HDF5 compatibility
            if labels.dtype.kind == 'U':  # If Unicode strings
                # Convert labels to ASCII (h5py compatible strings)
                ascii_labels = np.array([label.encode('ascii', 'ignore') for label in labels])
                h5f.create_dataset('labels', data=ascii_labels, dtype=h5py.special_dtype(vlen=str))
            else:
                h5f.create_dataset('labels', data=labels)
                
        logging.info(f"Dataset saved to {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving H5 file: {e}")
        return False

# Load dataset from H5 file
def load_from_h5(file_path):
    """Load features and labels from H5 file."""
    try:
        with h5py.File(file_path, 'r') as h5f:
            features = h5f['features'][:]
            labels_raw = h5f['labels'][:]
            
            # Convert byte strings back to Python strings if needed
            if isinstance(labels_raw[0], bytes):
                labels = np.array([label.decode('utf-8') for label in labels_raw])
            else:
                labels = labels_raw
                
        logging.info(f"Dataset loaded from {file_path}")
        return features, labels
    except Exception as e:
        logging.error(f"Error loading H5 file: {e}")
        return None, None

# Encode labels to integers
def encode_labels(labels):
    """Encode string labels to integers."""
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder

# Resize and normalize image
def preprocess_image(image, target_size=(128, 128)):
    """Resize image to target size and normalize pixel values."""
    if image is None:
        return None
    
    try:
        # Resize image
        resized = cv2.resize(image, target_size)
        
        # Convert to grayscale if it's not already
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
            
        # Normalize pixel values to [0, 1]
        normalized = gray / 255.0
        
        return normalized
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        return None
