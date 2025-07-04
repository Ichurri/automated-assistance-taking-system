import os
import cv2
import logging
import numpy as np
from skimage.feature import hog

from src.utils import preprocess_image
from src.extract import FeatureExtractor

class FaceRecognizer:
    """
    Module for real-time face recognition in video streams.
    """
    
    def __init__(self, model_instance, min_confidence=0.6, min_face_size=(30, 30)):
        """
        Initialize the face recognizer.
        
        Args:
            model_instance: Trained FaceRecognitionModel instance
            min_confidence (float): Minimum confidence threshold for recognition
            min_face_size (tuple): Minimum face size to detect
        """
        self.model = model_instance
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        
        # Load face detection cascade
        # Try multiple methods to find the cascade file
        cascade_file = None
        
        # Method 1: Try using cv2.data if available (OpenCV 4.x)
        if hasattr(cv2, 'data'):
            cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Method 2: Try standard OpenCV installation paths
        if cascade_file is None or not os.path.exists(cascade_file):
            possible_paths = [
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    cascade_file = path
                    break
        
        # Method 3: Try using pkg_resources if available
        if cascade_file is None or not os.path.exists(cascade_file):
            try:
                import pkg_resources
                cascade_file = pkg_resources.resource_filename('cv2', 'data/haarcascade_frontalface_default.xml')
            except:
                pass
        
        # Method 4: Last resort - try relative to cv2 module location
        if cascade_file is None or not os.path.exists(cascade_file):
            try:
                cv2_path = os.path.dirname(cv2.__file__)
                cascade_file = os.path.join(cv2_path, 'data', 'haarcascade_frontalface_default.xml')
            except:
                pass
        
        # Load the cascade classifier
        if cascade_file and os.path.exists(cascade_file):
            self.face_cascade = cv2.CascadeClassifier(cascade_file)
        else:
            # If all methods fail, try loading without path (OpenCV might find it automatically)
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        if self.face_cascade.empty():
            error_msg = "Error: Could not load face cascade classifier. Please ensure OpenCV is properly installed with Haar cascades."
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        # Initialize feature extractor (for HOG)
        self.feature_extractor = FeatureExtractor()
    
    def detect_and_recognize(self, frame):
        """
        Detect faces in a frame and recognize them.
        
        Args:
            frame: Video frame to process
            
        Returns:
            List of (x, y, w, h, label, confidence) tuples for recognized faces
        """
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size
        )
        
        recognized_faces = []
        
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face_img = frame[y:y+h, x:x+w]
            processed_face = preprocess_image(face_img)
            
            if processed_face is None:
                continue
                
            # Extract HOG features
            hog_features = self.feature_extractor.extract_hog_features(processed_face)
            
            if hog_features is None:
                continue
                
            # Predict identity
            label, confidence = self.model.predict(hog_features)
            
            if confidence >= self.min_confidence:
                recognized_faces.append((x, y, w, h, label, confidence))
            else:
                recognized_faces.append((x, y, w, h, "Unknown", confidence))
        
        return recognized_faces
    
    def run_recognition(self, camera_id=0, quit_key='q', window_name='Face Recognition'):
        """
        Run real-time face recognition on video from a camera.
        
        Args:
            camera_id (int): Camera device ID
            quit_key (str): Key to press to quit
            window_name (str): Name of the display window
            
        Returns:
            None
        """
        # Open video capture
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logging.error("Error: Could not open camera")
            return
            
        logging.info(f"Starting real-time face recognition. Press '{quit_key}' to quit.")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logging.error("Error: Failed to capture frame")
                break
                
            # Mirror the frame for more intuitive display
            frame = cv2.flip(frame, 1)
            
            # Detect and recognize faces
            faces = self.detect_and_recognize(frame)
            
            # Draw results on frame
            for (x, y, w, h, label, confidence) in faces:
                # Draw rectangle around face
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label with confidence
                text = f"{label} ({confidence:.2f})"
                cv2.putText(frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display the frame
            cv2.imshow(window_name, frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord(quit_key):
                break
                
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    def process_video_file(self, video_path, output_path=None):
        """
        Process a video file and recognize faces in it.
        
        Args:
            video_path (str): Path to the input video file
            output_path (str): Path to save the output video (None for no saving)
            
        Returns:
            None
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logging.error(f"Error: Could not open video file {video_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if needed
        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logging.info(f"Processing video with {frame_count} frames")
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % 50 == 0:
                logging.info(f"Processing frame {frame_idx}/{frame_count}")
            
            # Detect and recognize faces
            faces = self.detect_and_recognize(frame)
            
            # Draw results on frame
            for (x, y, w, h, label, confidence) in faces:
                # Draw rectangle around face
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label with confidence
                text = f"{label} ({confidence:.2f})"
                cv2.putText(frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Save frame if needed
            if writer is not None:
                writer.write(frame)
                
            # Display progress (optional)
            cv2.imshow('Processing Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        
        if output_path is not None:
            logging.info(f"Processed video saved to {output_path}")
    
    def recognize_image(self, image_path):
        """
        Detect and recognize faces in a static image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (image with annotations, list of recognized faces)
        """
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            logging.error(f"Error: Could not read image from {image_path}")
            return None, []
        
        # Get recognized faces
        recognized_faces = self.detect_and_recognize(frame)
        
        # Draw bounding boxes and labels on the image
        annotated_frame = frame.copy()
        for (x, y, w, h, label, confidence) in recognized_faces:
            # Draw rectangle around face
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Prepare label text with confidence
            text = f"{label} ({confidence:.2f})"
            
            # Draw filled rectangle for text background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated_frame, 
                         (x, y-text_size[1]-10), 
                         (x+text_size[0], y), 
                         (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(annotated_frame, text, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_frame, recognized_faces
