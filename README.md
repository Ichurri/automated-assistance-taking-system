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

The system includes a fully-featured GUI that makes all operations easy and accessible. To start the GUI:

```
python run_app.py
```

The GUI includes several tabs:

1. **Dashboard**: Overview of system status and quick actions
2. **Capture**: Capture new face images
3. **Dataset**: Manage captured face images
4. **Training**: Train the face recognition model
5. **Recognition**: Perform real-time face recognition
6. **Photo Recognition**: Process static photos
7. **Courses**: Manage courses, schedules, and student assignments
8. **Attendance**: View and export attendance records

## Course Management

The system includes comprehensive course management functionality that integrates with face recognition for attendance tracking:

### Key Features

- **Course Management**: Create, edit and delete courses with schedules and weekday information
- **Student Assignment**: Assign captured faces/students to specific courses
- **Course-Specific Attendance**: Take attendance for specific courses, with validation for enrolled students
- **Photo-Based Attendance**: Upload class photos to record attendance
- **Attendance Filtering**: Filter attendance records by course, date, and source (camera/photo)

### Using Course Management

1. **Creating Courses**:
   - Navigate to the Courses tab
   - Click "Add Course" and enter course details
   - Specify schedule and select weekdays

2. **Assigning Students**:
   - Select a course from the course list
   - Click "Add Student" to assign students from the face dataset

3. **Taking Course Attendance**:
   - **Camera Method**: Select a course and click "Take Attendance"
   - **Photo Method**: Select a course and click "Photo Attendance", then upload a class photo

4. **Viewing Attendance**:
   - Navigate to the Attendance tab
   - Use filters to view attendance by course, date, and source
   - Export attendance records to CSV

### Features

1. **Course Registration**
   - Create courses with name, schedule, and days of the week
   - Edit existing course details
   - Delete courses when no longer needed

2. **Student Assignment**
   - Assign students from the face database to specific courses
   - Remove students from courses
   - View enrolled students for each course

3. **Course-Specific Attendance**
   - Take attendance for specific courses
   - System will only record attendance for students enrolled in the course
   - Visual indicators show when recognized faces are not enrolled

4. **Attendance Filtering**
   - Filter attendance records by course and date
   - Export filtered attendance data to CSV

### Usage Guide

#### Creating a Course

1. Navigate to the **Courses** tab
2. Click "Add Course"
3. Enter the course name, schedule (format: HH:MM-HH:MM), and select days
4. Click "Save"

#### Assigning Students to a Course

1. Select a course from the list
2. Click "Add Student" to see available faces from the database
3. Select a student and click "Add"

#### Taking Course-Specific Attendance

1. Select a course from the list
2. Click "Take Attendance"
3. This will switch to the Recognition tab with the course context active
4. The system will only record attendance for students enrolled in the selected course
5. Non-enrolled students will be highlighted in orange

#### Viewing Attendance Records

1. Navigate to the **Attendance** tab
2. Use the course and date filters to view specific records
3. Click "Export" to save the filtered data as a CSV file

## Photo-Based Attendance

The system supports attendance tracking through uploaded photos, allowing instructors to take attendance from class photos even when real-time recognition isn't feasible.

### Features

1. **Photo Upload and Processing**
   - Upload class photos in common formats (JPG, PNG)
   - Process photos to detect and recognize multiple faces simultaneously
   - Visual indicators show recognition results directly on the photo

2. **Course Integration**
   - Select a specific course for photo attendance
   - System validates that recognized students are enrolled in the course
   - Non-enrolled students are identified but not marked as present

3. **Source Tracking**
   - Attendance records track the source (camera/photo)
   - Filter attendance records by source
   - Export includes source information

### Usage Guide

#### Taking Photo Attendance

**Method 1: From Course Tab**
1. Select a course from the list
2. Click "Photo Attendance" button
3. System will switch to Photo Recognition tab with the course pre-selected
4. Upload a class photo using "Upload Photo"
5. Process the photo using "Process Photo"
6. Click "Take Course Attendance" to record attendance

**Method 2: From Photo Recognition Tab**
1. Navigate to the Photo Recognition tab
2. Select a course from the dropdown
3. Upload and process a photo
4. Click "Take Course Attendance"

The system will provide a summary of:
- Students present (enrolled and recognized)
- Students detected but not enrolled (recognized but not in the course)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

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
