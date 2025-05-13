import os
import argparse
import logging
from src.utils import setup_logger
from src.capture import FaceCapture
from src.extract import FeatureExtractor
from src.train import FaceRecognitionModel
from src.recognize import FaceRecognizer

def main():
    """
    Main entry point for the face recognition system.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Face Recognition System')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['capture', 'extract', 'train', 'recognize'],
                       help='Operation mode: capture, extract, train, or recognize')
      # Capture mode arguments
    parser.add_argument('--name', type=str, 
                       help='Name of the person for dataset creation (required for capture mode)')
    parser.add_argument('--samples', type=int, default=20,
                       help='Number of samples to capture (default: 20)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--capture_interval', type=float, default=2.0,
                       help='Time interval between captures in seconds (default: 2.0)')
    parser.add_argument('--video', type=str,
                       help='Path to video file for face extraction')
    
    # Training mode arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    
    # Recognition mode arguments
    parser.add_argument('--confidence', type=float, default=0.6,
                       help='Minimum confidence threshold for recognition (default: 0.6)')
    parser.add_argument('--input_video', type=str,
                       help='Path to input video file for recognition')
    parser.add_argument('--output_video', type=str,
                       help='Path to output video file with recognition results')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger()
    
    # Execute requested mode
    if args.mode == 'capture':
        # Check required arguments
        if args.name is None and args.video is None:
            parser.error("Capture mode requires --name or --video")
        
        logger.info("Starting face capture mode")
        
        # Initialize face capture module
        face_capture = FaceCapture()
        
        if args.video:
            # Extract faces from video
            output_name = args.name or os.path.splitext(os.path.basename(args.video))[0]
            success = face_capture.capture_from_video(args.video, output_name)
            
            if success:
                logger.info(f"Successfully extracted faces from {args.video}")
            else:
                logger.error(f"Failed to extract faces from {args.video}")
        else:            # Capture from webcam
            success = face_capture.capture_face_dataset(
                args.name, 
                args.samples, 
                args.camera,
                args.capture_interval
            )
            
            if success:
                logger.info(f"Successfully captured dataset for {args.name}")
            else:
                logger.error(f"Failed to capture dataset for {args.name}")
    
    elif args.mode == 'extract':
        logger.info("Starting feature extraction mode")
        
        # Initialize feature extraction module
        feature_extractor = FeatureExtractor()
        
        # Process dataset
        success = feature_extractor.process_dataset()
        
        if success:
            logger.info("Successfully extracted HOG features from dataset")
        else:
            logger.error("Failed to extract HOG features from dataset")
    
    elif args.mode == 'train':
        logger.info("Starting model training mode")
        
        # Path to extracted features
        features_file = os.path.join('data', 'processed', 'face_features.h5')
        
        if not os.path.exists(features_file):
            logger.error(f"Features file not found: {features_file}")
            logger.error("Please run feature extraction first")
            return
        
        # Initialize and train model
        model = FaceRecognitionModel()
        history = model.train(features_file, epochs=args.epochs, batch_size=args.batch_size)
        
        if history:
            logger.info("Successfully trained face recognition model")
        else:
            logger.error("Failed to train face recognition model")
    
    elif args.mode == 'recognize':
        logger.info("Starting face recognition mode")
        
        # Initialize model
        model = FaceRecognitionModel()
        
        # Load trained model
        success = model.load_model()
        
        if not success:
            logger.error("Failed to load trained model")
            logger.error("Please train the model first")
            return
        
        # Initialize face recognizer
        recognizer = FaceRecognizer(model, min_confidence=args.confidence)
        
        if args.input_video:
            # Process video file
            recognizer.process_video_file(args.input_video, args.output_video)
        else:
            # Run real-time recognition
            recognizer.run_recognition(camera_id=args.camera)

if __name__ == '__main__':
    main()
