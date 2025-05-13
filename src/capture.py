import os
import time
import cv2
import logging
import numpy as np
from src.utils import ensure_directory, preprocess_image

class FaceCapture:
    """
    Module to detect faces in a video stream and capture images for dataset creation.
    """
    
    def __init__(self, output_dir='data/raw', min_face_size=(30, 30)):
        """
        Initialize the face capture module.
        
        Args:
            output_dir (str): Directory to save captured face images
            min_face_size (tuple): Minimum face size to detect
        """
        self.output_dir = output_dir
        self.min_face_size = min_face_size
        
        # Load face detection cascade
        cascades_dir = os.path.join(cv2.__path__[0], 'data')
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascades_dir, 'haarcascade_frontalface_default.xml')
        )
        
        if self.face_cascade.empty():
            error_msg = "Error: Could not load face cascade classifier"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # Ensure output directory exists
        ensure_directory(self.output_dir)
        
    def detect_faces(self, frame):
        """
        Detect faces in a frame.
        
        Args:
            frame: Video frame to detect faces in
            
        Returns:
            List of (x, y, w, h) tuples for detected faces
        """
        try:
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.min_face_size
            )
            
            return faces
        except Exception as e:
            logging.error(f"Error detecting faces: {e}")
            return []
    def capture_face_dataset(self, person_name, num_samples=20, camera_id=0, capture_interval=2.0):
        """
        Capture face images to create a dataset for a person automatically at regular intervals.
        
        Args:
            person_name (str): Name of the person to create dataset for
            num_samples (int): Number of face images to capture
            camera_id (int): Camera device ID
            capture_interval (float): Time interval between captures in seconds
            
        Returns:
            bool: True if dataset was successfully created, False otherwise
        """
        # Create directory for this person
        person_dir = os.path.join(self.output_dir, person_name)
        ensure_directory(person_dir)
        
        # Open video capture
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logging.error("Error: Could not open camera")
            return False
        
        logging.info(f"Starting automatic face capture for {person_name}. Press 'q' to quit.")
        
        sample_count = 0
        last_capture_time = time.time() - capture_interval  # Initialize to ensure first frame captures immediately
        
        while sample_count < num_samples:
            ret, frame = cap.read()
            
            if not ret:
                logging.error("Error: Failed to capture frame")
                break
            
            # Mirror the frame for more intuitive display
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display status and instructions
            cv2.putText(frame, f"Samples: {sample_count}/{num_samples}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            next_capture_in = max(0, capture_interval - (time.time() - last_capture_time))
            cv2.putText(frame, f"Next capture in: {next_capture_in:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, "Press 'q' to quit", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Face Capture', frame)
            
            # Automatic capture based on time interval
            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                # Only capture if a face is detected
                if len(faces) > 0:
                    # Get the largest face (by area)
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    
                    # Extract and preprocess the face
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Save the face image
                    img_path = os.path.join(person_dir, f"{person_name}_{sample_count}.jpg")
                    cv2.imwrite(img_path, face_img)
                    
                    logging.info(f"Captured sample {sample_count+1}/{num_samples}")
                    sample_count += 1
                    
                    # Update last capture time
                    last_capture_time = current_time
                else:
                    logging.warning("No face detected! Please position your face in frame.")
                    # Update last capture time to avoid rapid consecutive warnings
                    last_capture_time = current_time
            
            # Handle key presses for quitting
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        if sample_count < num_samples:
            logging.warning(f"Capture interrupted. Only {sample_count}/{num_samples} samples captured.")
            
        return sample_count > 0
    
    def capture_from_video(self, video_path, output_name, interval_frames=30):
        """
        Extract faces from a video file at regular intervals.
        
        Args:
            video_path (str): Path to the video file
            output_name (str): Name for the output directory/person
            interval_frames (int): Capture a face every N frames
            
        Returns:
            bool: True if faces were extracted, False otherwise
        """
        # Create directory for this dataset
        person_dir = os.path.join(self.output_dir, output_name)
        ensure_directory(person_dir)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logging.error(f"Error: Could not open video file {video_path}")
            return False
        
        frame_count = 0
        sample_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            frame_count += 1
            
            # Process every Nth frame
            if frame_count % interval_frames == 0:
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Save all detected faces
                for i, (x, y, w, h) in enumerate(faces):
                    # Extract the face
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Save the face image
                    img_path = os.path.join(person_dir, f"{output_name}_{sample_count}_{i}.jpg")
                    cv2.imwrite(img_path, face_img)
                    
                    sample_count += 1
        
        # Release resources
        cap.release()
        
        logging.info(f"Extracted {sample_count} faces from {video_path}")
        return sample_count > 0
