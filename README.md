# Face Recognition Attendance System

A Python-based face recognition system that uses Histogram of Oriented Gradients (HOG) for feature extraction and neural networks for classification. Designed for automated attendance tracking in educational settings.

## Overview

This system implements a complete face recognition pipeline:

1. **Face Capture**: Automatically detects and captures face images from a webcam or video feed.
2. **Feature Extraction**: Uses HOG descriptors to extract features from the collected face images.
3. **Model Training**: Trains a neural network on the HOG-extracted features to identify individuals.
4. **Real-Time Recognition**: Implements real-time face recognition on video input.
5. **Attendance Tracking**: Records when individuals are recognized in the video feed.

## Project Structure

```
automated-system-assistance-taking/
├── data/
│   ├── processed/    # Processed datasets in .h5 format
│   └── raw/          # Raw image data
├── gui/              # Graphical user interface
│   ├── __init__.py   # GUI package initialization
│   └── app.py        # Main GUI application
├── logs/             # Log files
├── models/           # Trained model weights and configurations (.keras format)
├── notebooks/        # Jupyter notebooks for experimentation
├── src/              # Source code
│   ├── capture.py    # Face detection and capture module
│   ├── extract.py    # Feature extraction using HOG
│   ├── train.py      # Neural network training
│   ├── recognize.py  # Real-time face recognition
│   ├── utils.py      # Utility functions
│   └── main.py       # Command-line interface
└── attendance_app.py # GUI launcher script
```

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd automated-system-assistance-taking
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

#### 1. Dataset Creation

To create your face dataset:

```
python src/main.py --mode capture --name "person_name" --samples 20 --capture_interval 2.0
```

This will automatically capture 20 face images from your webcam at 2-second intervals and save them to the raw data folder.

#### 2. Feature Extraction

To extract HOG features from captured images:

```
python src/main.py --mode extract
```

This processes all raw images and creates a feature dataset in the processed folder.

#### 3. Model Training

To train the neural network:

```
python src/main.py --mode train --epochs 50 --batch_size 32
```

#### 4. Face Recognition

To run real-time face recognition:

```
python src/main.py --mode recognize
```

### Graphical User Interface

For a more user-friendly experience, you can use the GUI application:

```
python attendance_app.py
```

The GUI provides an intuitive interface for:

- **Dashboard**: Overview of system status and quick actions
- **Capture**: Collect face images for individuals
- **Dataset**: Manage your face image datasets
- **Training**: Configure and run model training
- **Recognition**: Run real-time face recognition
- **Attendance**: View and export attendance records

The attendance system automatically records when people are recognized and tracks their presence time.

## Dependencies

- OpenCV (cv2)
- TensorFlow
- NumPy
- scikit-learn
- dlib
- h5py
- matplotlib
- PIL

## License

[MIT License](LICENSE)
