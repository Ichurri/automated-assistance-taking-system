import os
import cv2
import logging
import numpy as np
from skimage.feature import hog
from src.utils import ensure_directory, preprocess_image, save_to_h5

class FeatureExtractor:
    """
    Module for extracting HOG features from face images.
    """
    
    def __init__(self, input_dir='data/raw', output_dir='data/processed', 
                 image_size=(128, 128), hog_params=None):
        """
        Initialize the feature extractor.
        
        Args:
            input_dir (str): Directory containing raw face images
            output_dir (str): Directory to save processed features
            image_size (tuple): Size to resize images to
            hog_params (dict): Parameters for HOG feature extraction
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.image_size = image_size
        
        # Default HOG parameters
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',
            'visualize': False
        }
        
        # Update with custom parameters if provided
        if hog_params is not None:
            self.hog_params.update(hog_params)
            
        # Ensure output directory exists
        ensure_directory(self.output_dir)
    
    def extract_hog_features(self, image):
        """
        Extract HOG features from an image.
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            HOG features as a 1D array
        """
        try:
            features = hog(image, 
                          orientations=self.hog_params['orientations'],
                          pixels_per_cell=self.hog_params['pixels_per_cell'],
                          cells_per_block=self.hog_params['cells_per_block'],
                          block_norm=self.hog_params['block_norm'],
                          visualize=self.hog_params['visualize'])
            
            return features
        except Exception as e:
            logging.error(f"Error extracting HOG features: {e}")
            return None
    
    def process_dataset(self, output_filename='face_features.h5'):
        """
        Process all images in the input directory and extract HOG features.
        
        Args:
            output_filename (str): Name of the H5 file to save features
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        features_list = []
        labels_list = []
        
        # Get list of subdirectories (persons)
        person_dirs = [d for d in os.listdir(self.input_dir) 
                     if os.path.isdir(os.path.join(self.input_dir, d))]
        
        if not person_dirs:
            logging.error(f"No person directories found in {self.input_dir}")
            return False
        
        logging.info(f"Found {len(person_dirs)} persons in dataset")
        
        for person_name in person_dirs:
            person_dir = os.path.join(self.input_dir, person_name)
            
            # Get all image files for this person
            image_files = [f for f in os.listdir(person_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                logging.warning(f"No images found for {person_name}")
                continue
                
            logging.info(f"Processing {len(image_files)} images for {person_name}")
            
            for img_file in image_files:
                img_path = os.path.join(person_dir, img_file)
                
                try:
                    # Read and preprocess image
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        logging.warning(f"Could not read image: {img_path}")
                        continue
                        
                    # Preprocess the image (resize, convert to grayscale, normalize)
                    processed_img = preprocess_image(image, self.image_size)
                    
                    if processed_img is None:
                        continue
                    
                    # Extract HOG features
                    hog_features = self.extract_hog_features(processed_img)
                    
                    if hog_features is not None:
                        # Add to lists
                        features_list.append(hog_features)
                        labels_list.append(person_name)
                        
                except Exception as e:
                    logging.error(f"Error processing {img_path}: {e}")
        
        # Check if we have features
        if not features_list:
            logging.error("No features extracted from dataset")
            return False
            
        # Convert lists to numpy arrays
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        # Save features and labels to H5 file
        output_path = os.path.join(self.output_dir, output_filename)
        success = save_to_h5(features_array, labels_array, output_path)
        
        if success:
            logging.info(f"Processed {len(features_list)} images from {len(person_dirs)} persons")
            logging.info(f"Features shape: {features_array.shape}")
            
        return success
