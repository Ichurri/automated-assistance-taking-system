import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

# Add the parent directory to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import from our src modules
from src.capture import FaceCapture
from src.extract import FeatureExtractor
from src.train import FaceRecognitionModel
from src.recognize import FaceRecognizer
from src.utils import setup_logger

class AttendanceSystemApp:
    """Main application class for the Face Recognition Attendance System GUI."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1200x700")
        self.root.minsize(1000, 600)
        
        # Set up logger
        self.logger = setup_logger()
        
        # Initialize state variables
        self.camera_id = 0
        self.cap = None
        self.is_capturing = False
        self.recognition_running = False
        self.frame_thread = None
        self.person_name = ""
        self.current_person = None
        self.attendance_records = {}
        
        # Course management variables
        self.courses = {}  # Dictionary to store course information
        self.current_course = None  # Currently selected course
        self.current_attendance_context = None  # Context for current attendance session
        self.load_courses()  # Load saved courses if available
        
        # Initialize modules
        self.face_capture = FaceCapture()
        self.feature_extractor = FeatureExtractor()
        self.model = FaceRecognitionModel()
        self.recognizer = None
        
        # Create UI elements
        self.create_ui()
        
        # When window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook (tabbed interface)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Dashboard
        dashboard_tab = ttk.Frame(notebook)
        notebook.add(dashboard_tab, text="Dashboard")
        self.setup_dashboard_tab(dashboard_tab)
        
        # Tab 2: Capture
        capture_tab = ttk.Frame(notebook)
        notebook.add(capture_tab, text="Capture")
        self.setup_capture_tab(capture_tab)
        
        # Tab 3: Dataset Management
        dataset_tab = ttk.Frame(notebook)
        notebook.add(dataset_tab, text="Dataset")
        self.setup_dataset_tab(dataset_tab)
        
        # Tab 4: Training
        training_tab = ttk.Frame(notebook)
        notebook.add(training_tab, text="Training")
        self.setup_training_tab(training_tab)
        
        # Tab 5: Recognition
        recognition_tab = ttk.Frame(notebook)
        notebook.add(recognition_tab, text="Recognition")
        self.setup_recognition_tab(recognition_tab)
        
        # Tab 6: Photo Recognition
        photo_tab = ttk.Frame(notebook)
        notebook.add(photo_tab, text="Photo Recognition")
        self.setup_photo_recognition_tab(photo_tab)
        
        # Tab 7: Course Management
        course_tab = ttk.Frame(notebook)
        notebook.add(course_tab, text="Courses")
        self.setup_course_tab(course_tab)
        
        # Tab 8: Attendance Records
        attendance_tab = ttk.Frame(notebook)
        notebook.add(attendance_tab, text="Attendance")
        self.setup_attendance_tab(attendance_tab)
        
        # Status bar
        status_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, padding=2)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(fill=tk.X)
    
    def setup_dashboard_tab(self, parent):
        """Set up the dashboard tab with system status and quick actions."""
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(frame, text="Face Recognition Attendance System", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)
        
        # System info
        info_frame = ttk.LabelFrame(frame, text="System Information", padding=10)
        info_frame.pack(fill=tk.X, pady=10)
        
        # Check if model is trained
        model_status = "Trained" if os.path.exists(os.path.join("models", "face_recognition_model.keras")) else "Not Trained"
        
        # Count number of persons and images in dataset
        persons = 0
        images = 0
        if os.path.exists(os.path.join("data", "raw")):
            person_dirs = [d for d in os.listdir(os.path.join("data", "raw")) 
                        if os.path.isdir(os.path.join("data", "raw", d))]
            persons = len(person_dirs)
            
            for person_dir in person_dirs:
                dir_path = os.path.join("data", "raw", person_dir)
                image_files = [f for f in os.listdir(dir_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                images += len(image_files)
        
        # Display info
        ttk.Label(info_frame, text=f"Model Status: {model_status}").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(info_frame, text=f"Persons in Dataset: {persons}").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(info_frame, text=f"Total Face Images: {images}").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(info_frame, text=f"Total Courses: {len(self.courses)}").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        
        # Course information section
        courses_frame = ttk.LabelFrame(frame, text="Course Information", padding=10)
        courses_frame.pack(fill=tk.X, pady=10)
        
        # Create a simple treeview to display course summary
        columns = ("name", "schedule", "days", "students")
        self.dashboard_course_tree = ttk.Treeview(courses_frame, columns=columns, show="headings", height=3)
        
        # Define headings
        self.dashboard_course_tree.heading("name", text="Course Name")
        self.dashboard_course_tree.heading("schedule", text="Schedule")
        self.dashboard_course_tree.heading("days", text="Days")
        self.dashboard_course_tree.heading("students", text="Students")
        
        # Define columns
        self.dashboard_course_tree.column("name", width=150)
        self.dashboard_course_tree.column("schedule", width=100)
        self.dashboard_course_tree.column("days", width=150)
        self.dashboard_course_tree.column("students", width=70)
        
        # Pack tree 
        self.dashboard_course_tree.pack(fill=tk.X, pady=5)
        
        # Add courses to treeview
        self.update_dashboard_courses()
        
        # Button to manage courses
        ttk.Button(courses_frame, text="Manage Courses", 
                  command=lambda: self.select_tab(6)).pack(anchor="e", pady=5)  # 6 is index of Courses tab
        
        # Quick actions
        actions_frame = ttk.LabelFrame(frame, text="Quick Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(actions_frame, text="Capture New Faces", 
                  command=lambda: self.select_tab(1)).grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Button(actions_frame, text="Process Dataset", 
                  command=self.process_dataset).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(actions_frame, text="Train Model", 
                  command=lambda: self.select_tab(3)).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(actions_frame, text="Start Recognition", 
                  command=lambda: self.select_tab(4)).grid(row=0, column=3, padx=5, pady=5)
        
        # Recent activity
        activity_frame = ttk.LabelFrame(frame, text="Recent Activity", padding=10)
        activity_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.activity_text = tk.Text(activity_frame, height=10, state="disabled")
        self.activity_text.pack(fill=tk.BOTH, expand=True)
        
        # Add some initial activity
        self.add_activity("System initialized")
    
    def setup_capture_tab(self, parent):
        """Set up the capture tab for capturing face images."""
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - controls
        left_panel = ttk.Frame(frame, padding=10, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Person info
        ttk.Label(left_panel, text="Person Information", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        info_frame = ttk.Frame(left_panel)
        info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(info_frame, text="Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(info_frame, textvariable=self.name_var)
        name_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        info_frame.columnconfigure(1, weight=1)
        
        # Capture settings
        settings_frame = ttk.LabelFrame(left_panel, text="Capture Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(settings_frame, text="Camera:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.camera_var = tk.IntVar(value=0)
        camera_entry = ttk.Combobox(settings_frame, textvariable=self.camera_var, values=[0, 1, 2, 3])
        camera_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Samples:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.samples_var = tk.IntVar(value=20)
        samples_entry = ttk.Entry(settings_frame, textvariable=self.samples_var)
        samples_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Interval (s):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.interval_var = tk.DoubleVar(value=2.0)
        interval_entry = ttk.Entry(settings_frame, textvariable=self.interval_var)
        interval_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        # Capture control buttons
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        self.start_capture_btn = ttk.Button(buttons_frame, text="Start Capture", command=self.start_capture)
        self.start_capture_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_capture_btn = ttk.Button(buttons_frame, text="Stop Capture", command=self.stop_capture, state="disabled")
        self.stop_capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(left_panel, text="Capture Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.StringVar(value="0/0")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(anchor="w", pady=5)
        
        self.progressbar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progressbar.pack(fill=tk.X, pady=5)
        
        # Right panel - video feed
        right_panel = ttk.Frame(frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video label
        self.video_label = ttk.Label(right_panel)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder for video
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder_img = Image.fromarray(cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB))
        placeholder_tk = ImageTk.PhotoImage(image=placeholder_img)
        self.video_label.configure(image=placeholder_tk)
        self.video_label.image = placeholder_tk
    
    def setup_dataset_tab(self, parent):
        """Set up the dataset management tab."""
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Top controls
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(top_frame, text="Dataset Management", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT)
        
        ttk.Button(top_frame, text="Refresh", command=self.refresh_dataset).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top_frame, text="Process Dataset", command=self.process_dataset).pack(side=tk.RIGHT, padx=5)
        
        # Split view
        paned_window = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Person list frame
        person_frame = ttk.LabelFrame(paned_window, text="Persons", padding=10)
        paned_window.add(person_frame, weight=1)
        
        # Person listbox with scrollbar
        person_scroll = ttk.Scrollbar(person_frame)
        person_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.person_listbox = tk.Listbox(person_frame)
        self.person_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.person_listbox.config(yscrollcommand=person_scroll.set)
        person_scroll.config(command=self.person_listbox.yview)
        
        # Person listbox selection event
        self.person_listbox.bind('<<ListboxSelect>>', self.on_person_select)
        
        # Image list frame
        image_frame = ttk.LabelFrame(paned_window, text="Images", padding=10)
        paned_window.add(image_frame, weight=2)
        
        # Image frame - top controls
        img_top = ttk.Frame(image_frame)
        img_top.pack(fill=tk.X, pady=(0, 10))
        
        self.selected_person_var = tk.StringVar(value="No person selected")
        ttk.Label(img_top, textvariable=self.selected_person_var).pack(side=tk.LEFT)
        
        ttk.Button(img_top, text="Delete Selected", command=self.delete_selected_images).pack(side=tk.RIGHT)
        
        # Image frame - grid for thumbnails
        self.thumbnail_frame = ttk.Frame(image_frame)
        self.thumbnail_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbar for thumbnails
        self.thumbnail_canvas = tk.Canvas(self.thumbnail_frame)
        thumb_scrollbar = ttk.Scrollbar(self.thumbnail_frame, orient="vertical", command=self.thumbnail_canvas.yview)
        self.thumbnail_canvas.configure(yscrollcommand=thumb_scrollbar.set)
        
        thumb_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.thumbnail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for thumbnails
        self.thumbnail_inner_frame = ttk.Frame(self.thumbnail_canvas)
        self.thumbnail_canvas_window = self.thumbnail_canvas.create_window((0, 0), window=self.thumbnail_inner_frame, anchor="nw")
        
        self.thumbnail_inner_frame.bind("<Configure>", self.on_thumbnail_frame_configure)
        self.thumbnail_canvas.bind("<Configure>", self.on_thumbnail_canvas_configure)
        
        # Populate the person list
        self.refresh_dataset()
    
    def setup_training_tab(self, parent):
        """Set up the training tab."""
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - training parameters
        left_panel = ttk.Frame(frame, padding=10, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        ttk.Label(left_panel, text="Training Parameters", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        params_frame = ttk.LabelFrame(left_panel, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=50)
        epochs_entry = ttk.Entry(params_frame, textvariable=self.epochs_var)
        epochs_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.batch_size_var = tk.IntVar(value=32)
        batch_entry = ttk.Entry(params_frame, textvariable=self.batch_size_var)
        batch_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(params_frame, text="Validation Split:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.val_split_var = tk.DoubleVar(value=0.2)
        val_entry = ttk.Entry(params_frame, textvariable=self.val_split_var)
        val_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        params_frame.columnconfigure(1, weight=1)
        
        # Training control
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.train_btn = ttk.Button(control_frame, text="Start Training", command=self.start_training)
        self.train_btn.pack(fill=tk.X, pady=5)
        
        # Training status
        status_frame = ttk.LabelFrame(left_panel, text="Training Status", padding=10)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.train_status_var = tk.StringVar(value="Not started")
        ttk.Label(status_frame, textvariable=self.train_status_var).pack(anchor="w", pady=5)
        
        self.train_progress = ttk.Progressbar(status_frame, orient="horizontal", mode="determinate")
        self.train_progress.pack(fill=tk.X, pady=5)
        
        # Right panel - training results
        right_panel = ttk.Frame(frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Training history visualization
        history_frame = ttk.LabelFrame(right_panel, text="Training History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        # Check if training history image exists
        self.history_label = ttk.Label(history_frame)
        self.history_label.pack(fill=tk.BOTH, expand=True)
        
        self.update_training_history()
    
    def setup_recognition_tab(self, parent):
        """Set up the recognition tab."""
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - recognition controls
        left_panel = ttk.Frame(frame, padding=10, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        ttk.Label(left_panel, text="Recognition Controls", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        settings_frame = ttk.LabelFrame(left_panel, text="Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(settings_frame, text="Camera:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.rec_camera_var = tk.IntVar(value=0)
        camera_entry = ttk.Combobox(settings_frame, textvariable=self.rec_camera_var, values=[0, 1, 2, 3])
        camera_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.confidence_var = tk.DoubleVar(value=0.6)
        confidence_entry = ttk.Entry(settings_frame, textvariable=self.confidence_var)
        confidence_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        settings_frame.columnconfigure(1, weight=1)
        
        # Control buttons
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        self.start_recognition_btn = ttk.Button(buttons_frame, text="Start Recognition", command=self.start_recognition)
        self.start_recognition_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_recognition_btn = ttk.Button(buttons_frame, text="Stop Recognition", command=self.stop_recognition, state="disabled")
        self.stop_recognition_btn.pack(side=tk.LEFT, padx=5)
        
        # Recognition stats
        stats_frame = ttk.LabelFrame(left_panel, text="Recognition Stats", padding=10)
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.current_person_var = tk.StringVar(value="No one detected")
        ttk.Label(stats_frame, text="Current Person:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(stats_frame, textvariable=self.current_person_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        self.confidence_level_var = tk.StringVar(value="0.00")
        ttk.Label(stats_frame, text="Confidence:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Label(stats_frame, textvariable=self.confidence_level_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Right panel - video feed
        right_panel = ttk.Frame(frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video label
        self.rec_video_label = ttk.Label(right_panel)
        self.rec_video_label.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder for video
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder_img = Image.fromarray(cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB))
        placeholder_tk = ImageTk.PhotoImage(image=placeholder_img)
        self.rec_video_label.configure(image=placeholder_tk)
        self.rec_video_label.image = placeholder_tk
    
    def setup_photo_recognition_tab(self, parent):
        """Set up the photo recognition tab."""
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - controls
        left_panel = ttk.Frame(frame, padding=10, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        ttk.Label(left_panel, text="Photo Recognition", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(left_panel, text="Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.photo_confidence_var = tk.DoubleVar(value=0.6)
        confidence_entry = ttk.Entry(settings_frame, textvariable=self.photo_confidence_var)
        confidence_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        settings_frame.columnconfigure(1, weight=1)
        
        # Control buttons
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        self.upload_photo_btn = ttk.Button(buttons_frame, text="Upload Photo", command=self.upload_photo)
        self.upload_photo_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.process_photo_btn = ttk.Button(buttons_frame, text="Process Photo", command=self.process_photo, state="disabled")
        self.process_photo_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.save_result_btn = ttk.Button(buttons_frame, text="Save Result", command=self.save_photo_result, state="disabled")
        self.save_result_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Recognition results
        results_frame = ttk.LabelFrame(left_panel, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.photo_results_text = tk.Text(results_frame, height=10, width=30, wrap=tk.WORD)
        self.photo_results_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - image display
        right_panel = ttk.Frame(frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.photo_canvas = tk.Canvas(right_panel, bg="black")
        self.photo_canvas.pack(fill=tk.BOTH, expand=True)
          # Store variables for photo recognition
        self.current_photo_path = None
        self.processed_photo = None
        self.photo_recognition_results = []
        
    def setup_attendance_tab(self, parent):
        """Set up the attendance records tab."""
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Top controls
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(top_frame, text="Attendance Records", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT)
        
        # Course selection
        course_frame = ttk.Frame(top_frame)
        course_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(course_frame, text="Filter by course:").pack(side=tk.LEFT, padx=(0, 5))
        self.attendance_course_var = tk.StringVar(value="All Courses")
        self.course_combobox = ttk.Combobox(course_frame, textvariable=self.attendance_course_var, width=20)
        self.course_combobox.pack(side=tk.LEFT)
        self.course_combobox.bind("<<ComboboxSelected>>", self.filter_attendance_by_course)
        
        # Date filter (optional)
        date_frame = ttk.Frame(top_frame)
        date_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(date_frame, text="Date:").pack(side=tk.LEFT, padx=(0, 5))
        self.attendance_date_var = tk.StringVar(value="All Dates")
        self.date_combobox = ttk.Combobox(date_frame, textvariable=self.attendance_date_var, width=15)
        self.date_combobox.pack(side=tk.LEFT)
        self.date_combobox.bind("<<ComboboxSelected>>", self.filter_attendance_by_date)
        
        # Export and clear buttons
        ttk.Button(top_frame, text="Export", command=self.export_attendance).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top_frame, text="Clear All", command=self.clear_attendance).pack(side=tk.RIGHT, padx=5)
        
        # Attendance treeview
        columns = ("name", "course", "date", "time_in", "time_out", "duration")
        self.attendance_tree = ttk.Treeview(frame, columns=columns, show="headings")
        
        # Define headings
        self.attendance_tree.heading("name", text="Name")
        self.attendance_tree.heading("course", text="Course")
        self.attendance_tree.heading("date", text="Date")
        self.attendance_tree.heading("time_in", text="Time In")
        self.attendance_tree.heading("time_out", text="Time Out")
        self.attendance_tree.heading("duration", text="Duration")
        
        # Define columns
        self.attendance_tree.column("name", width=150)
        self.attendance_tree.column("course", width=150)
        self.attendance_tree.column("date", width=100)
        self.attendance_tree.column("time_in", width=100)
        self.attendance_tree.column("time_out", width=100)
        self.attendance_tree.column("duration", width=80)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscroll=scrollbar.set)
        
        # Pack tree and scrollbar
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.attendance_tree.pack(fill=tk.BOTH, expand=True)
        
        # Update course dropdown
        self.update_attendance_course_filter()
    
    def setup_course_tab(self, parent):
        """Set up the course management tab."""
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a horizontal split with courses on left, students on right
        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # ---- Left frame: Course Management ----
        ttk.Label(left_frame, text="Course Management", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Course list frame
        course_list_frame = ttk.LabelFrame(left_frame, text="Courses", padding=10)
        course_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Course Treeview
        columns = ("name", "schedule", "days")
        self.course_tree = ttk.Treeview(course_list_frame, columns=columns, show="headings")
        
        # Define headings
        self.course_tree.heading("name", text="Course Name")
        self.course_tree.heading("schedule", text="Schedule")
        self.course_tree.heading("days", text="Days")
        
        # Define columns
        self.course_tree.column("name", width=150)
        self.course_tree.column("schedule", width=150)
        self.course_tree.column("days", width=150)
        
        # Add scrollbar
        course_scrollbar = ttk.Scrollbar(course_list_frame, orient=tk.VERTICAL, command=self.course_tree.yview)
        self.course_tree.configure(yscroll=course_scrollbar.set)
        
        # Bind selection event
        self.course_tree.bind("<<TreeviewSelect>>", self.on_course_selected)
        
        # Pack tree and scrollbar
        course_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.course_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Course control buttons
        course_buttons_frame = ttk.Frame(course_list_frame)
        course_buttons_frame.pack(fill=tk.X)
        
        ttk.Button(course_buttons_frame, text="Add Course", command=self.add_course_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(course_buttons_frame, text="Edit Course", command=self.edit_course_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(course_buttons_frame, text="Delete Course", command=self.delete_course).pack(side=tk.LEFT, padx=5)
        
        # ---- Right frame: Student Management ----
        ttk.Label(right_frame, text="Students in Selected Course", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Student list frame
        student_list_frame = ttk.LabelFrame(right_frame, text="Students", padding=10)
        student_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Student Treeview
        columns = ("name",)
        self.student_tree = ttk.Treeview(student_list_frame, columns=columns, show="headings")
        
        # Define heading
        self.student_tree.heading("name", text="Student Name")
        
        # Define column
        self.student_tree.column("name", width=200)
        
        # Add scrollbar
        student_scrollbar = ttk.Scrollbar(student_list_frame, orient=tk.VERTICAL, command=self.student_tree.yview)
        self.student_tree.configure(yscroll=student_scrollbar.set)
        
        # Pack tree and scrollbar
        student_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.student_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Student control buttons
        student_buttons_frame = ttk.Frame(student_list_frame)
        student_buttons_frame.pack(fill=tk.X)
        
        ttk.Button(student_buttons_frame, text="Add Student", command=self.add_student_to_course).pack(side=tk.LEFT, padx=5)
        ttk.Button(student_buttons_frame, text="Remove Student", command=self.remove_student_from_course).pack(side=tk.LEFT, padx=5)
        ttk.Button(student_buttons_frame, text="Take Attendance", command=self.take_course_attendance).pack(side=tk.LEFT, padx=5)
        
        # Initialize course list
        self.update_course_list()
    
    def select_tab(self, tab_idx):
        """Select a specific tab in the notebook."""
        notebook = self.root.winfo_children()[0].winfo_children()[0]  # Get the notebook widget
        notebook.select(tab_idx)
        
        # Reset attendance context when switching away from recognition tab
        if tab_idx != 4:  # 4 is the index of Recognition tab
            self.current_attendance_context = None
    
    def update_frame(self):
        """Update video frame in a separate thread."""
        while self.is_capturing or self.recognition_running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Flip the frame for more intuitive display
                    frame = cv2.flip(frame, 1)
                    
                    if self.is_capturing:
                        # For face capture tab
                        # Detect faces
                        faces = self.face_capture.detect_faces(frame)
                        
                        # Draw rectangles around faces
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Convert to PhotoImage
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        imgtk = ImageTk.PhotoImage(image=img)
                        
                        # Update video label
                        self.video_label.configure(image=imgtk)
                        self.video_label.image = imgtk
                    
                    elif self.recognition_running:
                        # For recognition tab
                        # Add course info to frame if in course context
                        if self.current_attendance_context:
                            course_name = self.current_attendance_context['course_name']
                            cv2.putText(frame, f"Course: {course_name}", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Detect and recognize faces
                        if self.recognizer:
                            faces = self.recognizer.detect_and_recognize(frame)
                            
                            # Draw results on frame
                            for (x, y, w, h, label, confidence) in faces:
                                # Draw rectangle around face
                                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                                
                                # Check if in course mode and person is enrolled in this course
                                if self.current_attendance_context and label != "Unknown":
                                    course_id = self.current_attendance_context['course_id']
                                    if course_id in self.courses:
                                        students = self.courses[course_id].get('students', [])
                                        if label not in students:
                                            color = (0, 165, 255)  # Orange for not enrolled
                                
                                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                                
                                # Draw label with confidence
                                text = f"{label} ({confidence:.2f})"
                                cv2.putText(frame, text, (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                
                                # Update recognition stats
                                if confidence > float(self.confidence_var.get()):
                                    self.current_person_var.set(label)
                                    self.confidence_level_var.set(f"{confidence:.2f}")
                                    
                                    # Update attendance record
                                    self.update_attendance(label)
                            
                            # Convert to PhotoImage
                            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            imgtk = ImageTk.PhotoImage(image=img)
                            
                            # Update video label
                            self.rec_video_label.configure(image=imgtk)
                            self.rec_video_label.image = imgtk
            
            # Sleep to reduce CPU usage
            time.sleep(0.03)
    
    def start_capture(self):
        """Start face capture process."""
        person_name = self.name_var.get().strip()
        
        if not person_name:
            messagebox.showerror("Error", "Please enter a person name")
            return
        
        try:
            samples = int(self.samples_var.get())
            camera_id = int(self.camera_var.get())
            interval = float(self.interval_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid input values")
            return
        
        # Update UI
        self.start_capture_btn.config(state="disabled")
        self.stop_capture_btn.config(state="normal")
        self.progress_var.set(f"0/{samples}")
        self.progressbar["maximum"] = samples
        self.progressbar["value"] = 0
        
        # Create directory for this person
        person_dir = os.path.join("data", "raw", person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        # Start video capture
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.stop_capture()
            return
        
        # Set capturing flag
        self.is_capturing = True
        self.person_name = person_name
        
        # Start frame update thread
        self.frame_thread = threading.Thread(target=self.update_frame)
        self.frame_thread.daemon = True
        self.frame_thread.start()
        
        # Start auto-capture thread
        self.capture_thread = threading.Thread(target=self.auto_capture, 
                                               args=(person_name, samples, interval))
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.status_var.set(f"Capturing faces for {person_name}...")
        self.add_activity(f"Started face capture for {person_name}")
    
    def auto_capture(self, person_name, num_samples, interval):
        """Automatically capture face images at regular intervals."""
        person_dir = os.path.join("data", "raw", person_name)
        sample_count = 0
        last_capture_time = time.time() - interval  # Start immediately
        
        while self.is_capturing and sample_count < num_samples:
            current_time = time.time()
            
            if current_time - last_capture_time >= interval:
                # Get the current frame
                if self.cap is not None and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret:
                        # Flip the frame
                        frame = cv2.flip(frame, 1)
                        
                        # Detect faces
                        faces = self.face_capture.detect_faces(frame)
                        
                        # Only capture if a face is detected
                        if len(faces) > 0:
                            # Get the largest face (by area)
                            largest_face = max(faces, key=lambda f: f[2] * f[3])
                            x, y, w, h = largest_face
                            
                            # Extract the face
                            face_img = frame[y:y+h, x:x+w]
                            
                            # Save the face image
                            img_path = os.path.join(person_dir, f"{person_name}_{sample_count}.jpg")
                            cv2.imwrite(img_path, face_img)
                            
                            # Update counter and UI
                            sample_count += 1
                            
                            # Update progress in main thread
                            self.root.after(0, self.update_progress, sample_count, num_samples)
                            
                            # Update last capture time
                            last_capture_time = current_time
                        else:
                            self.root.after(0, self.status_var.set, "No face detected! Please position your face in frame.")
                            
                            # Update last capture time to avoid rapid consecutive warnings
                            last_capture_time = current_time
            
            # Sleep to reduce CPU usage
            time.sleep(0.1)
        
        # Stop capture when done
        if sample_count >= num_samples:
            self.root.after(0, self.capture_complete)
    
    def update_progress(self, current, total):
        """Update the progress bar and text."""
        self.progress_var.set(f"{current}/{total}")
        self.progressbar["value"] = current
    
    def capture_complete(self):
        """Handle completion of the capture process."""
        self.stop_capture()
        messagebox.showinfo("Capture Complete", f"Successfully captured faces for {self.person_name}")
        self.add_activity(f"Completed face capture for {self.person_name}")
        
        # Refresh dataset tab
        self.refresh_dataset()
    
    def stop_capture(self):
        """Stop face capture process."""
        self.is_capturing = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Update UI
        self.start_capture_btn.config(state="normal")
        self.stop_capture_btn.config(state="disabled")
        self.status_var.set("Capture stopped")
    
    def refresh_dataset(self):
        """Refresh the dataset view."""
        # Clear current list
        self.person_listbox.delete(0, tk.END)
        
        # Get list of person directories
        raw_dir = os.path.join("data", "raw")
        if os.path.exists(raw_dir):
            person_dirs = [d for d in os.listdir(raw_dir) 
                          if os.path.isdir(os.path.join(raw_dir, d))]
            
            for person in person_dirs:
                self.person_listbox.insert(tk.END, person)
        
        # Clear current person
        self.current_person = None
        self.selected_person_var.set("No person selected")
        
        # Clear thumbnails
        for widget in self.thumbnail_inner_frame.winfo_children():
            widget.destroy()
    
    def on_person_select(self, event):
        """Handle selection of a person in the listbox."""
        selection = self.person_listbox.curselection()
        if selection:
            # Get selected person
            person = self.person_listbox.get(selection[0])
            self.current_person = person
            self.selected_person_var.set(f"Selected: {person}")
            
            # Load person's images
            self.load_person_images(person)
    
    def load_person_images(self, person):
        """Load and display thumbnails for a person's images."""
        # Clear current thumbnails
        for widget in self.thumbnail_inner_frame.winfo_children():
            widget.destroy()
        
        # Get person's image files
        person_dir = os.path.join("data", "raw", person)
        image_files = [f for f in os.listdir(person_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Thumbnail size
        thumb_size = (100, 100)
        
        # Grid layout
        num_cols = 4
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(person_dir, img_file)
            
            # Create a frame for the thumbnail and checkbox
            thumb_frame = ttk.Frame(self.thumbnail_inner_frame)
            thumb_frame.grid(row=i // num_cols, column=i % num_cols, padx=5, pady=5)
            
            # Load image and create thumbnail
            try:
                img = Image.open(img_path)
                img.thumbnail(thumb_size)
                photo = ImageTk.PhotoImage(img)
                
                # Create label for thumbnail
                label = ttk.Label(thumb_frame, image=photo)
                label.image = photo  # Keep a reference
                label.pack()
                
                # Add filename label
                ttk.Label(thumb_frame, text=img_file).pack()
                
                # Add a checkbox for selection
                var = tk.BooleanVar()
                check = ttk.Checkbutton(thumb_frame, variable=var)
                check.pack()
                
                # Store the path and var for later use
                thumb_frame.img_path = img_path
                thumb_frame.img_var = var
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    def on_thumbnail_frame_configure(self, event):
        """Handle resize of thumbnail inner frame."""
        self.thumbnail_canvas.configure(scrollregion=self.thumbnail_canvas.bbox("all"))
    
    def on_thumbnail_canvas_configure(self, event):
        """Handle resize of thumbnail canvas."""
        self.thumbnail_canvas.itemconfig(self.thumbnail_canvas_window, width=event.width)
    
    def delete_selected_images(self):
        """Delete selected images from the dataset."""
        if not self.current_person:
            messagebox.showinfo("Info", "No person selected")
            return
        
        # Collect selected images
        selected_images = []
        for widget in self.thumbnail_inner_frame.winfo_children():
            if hasattr(widget, 'img_path') and hasattr(widget, 'img_var'):
                if widget.img_var.get():
                    selected_images.append(widget.img_path)
        
        if not selected_images:
            messagebox.showinfo("Info", "No images selected")
            return
        
        # Confirm deletion
        if messagebox.askyesno("Confirm", f"Delete {len(selected_images)} selected images?"):
            # Delete files
            for img_path in selected_images:
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"Error deleting {img_path}: {e}")
            
            # Reload images
            self.load_person_images(self.current_person)
            self.add_activity(f"Deleted {len(selected_images)} images for {self.current_person}")
    
    def process_dataset(self):
        """Process the dataset to extract HOG features."""
        # Check if raw data exists
        raw_dir = os.path.join("data", "raw")
        if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
            messagebox.showinfo("Info", "No raw data available. Please capture faces first.")
            return
        
        # Confirm processing
        if messagebox.askyesno("Confirm", "Process the dataset and extract HOG features?"):
            self.status_var.set("Processing dataset...")
            
            # Process in a separate thread to keep UI responsive
            processing_thread = threading.Thread(target=self.run_processing)
            processing_thread.daemon = True
            processing_thread.start()
    
    def run_processing(self):
        """Run the dataset processing in a background thread."""
        try:
            # Process dataset
            success = self.feature_extractor.process_dataset()
            
            # Update UI
            if success:
                self.root.after(0, self.process_complete, True)
            else:
                self.root.after(0, self.process_complete, False)
                
        except Exception as e:
            print(f"Error processing dataset: {e}")
            self.root.after(0, self.process_complete, False, str(e))
    
    def process_complete(self, success, error=None):
        """Handle completion of dataset processing."""
        if success:
            self.status_var.set("Dataset processing complete")
            messagebox.showinfo("Success", "Dataset processing complete. HOG features extracted.")
            self.add_activity("HOG features extracted from dataset")
        else:
            self.status_var.set("Dataset processing failed")
            error_msg = f"Error: {error}" if error else "Unknown error"
            messagebox.showerror("Error", f"Dataset processing failed. {error_msg}")
            self.add_activity("Dataset processing failed")
    
    def update_training_history(self):
        """Update the training history visualization."""
        history_path = os.path.join("models", "training_history.png")
        
        if os.path.exists(history_path):
            try:
                img = Image.open(history_path)
                img = img.resize((800, 400), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                self.history_label.configure(image=photo)
                self.history_label.image = photo
            except Exception as e:
                print(f"Error loading training history: {e}")
        else:
            # Display placeholder text
            self.history_label.configure(image=None)
            self.history_label.configure(text="No training history available")
    
    def start_training(self):
        """Start the model training process."""
        # Check if processed data exists
        features_file = os.path.join("data", "processed", "face_features.h5")
        if not os.path.exists(features_file):
            messagebox.showinfo("Info", "No processed data available. Please process the dataset first.")
            return
        
        # Get training parameters
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            validation_split = float(self.val_split_var.get())
            
            if epochs <= 0 or batch_size <= 0 or validation_split <= 0 or validation_split >= 1:
                raise ValueError("Invalid parameter values")
                
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values")
            return
        
        # Disable training button
        self.train_btn.config(state="disabled")
        self.train_status_var.set("Initializing training...")
        self.train_progress["value"] = 0
        
        # Train in a separate thread
        training_thread = threading.Thread(target=self.run_training, 
                                          args=(features_file, epochs, batch_size, validation_split))
        training_thread.daemon = True
        training_thread.start()
    
    def run_training(self, features_file, epochs, batch_size, validation_split):
        """Run model training in a background thread."""
        try:
            # Initialize model
            self.model = FaceRecognitionModel()
            
            # Add a custom callback to update progress
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, app, epochs):
                    self.app = app
                    self.epochs = epochs
                
                def on_epoch_begin(self, epoch, logs=None):
                    # Update status in main thread
                    self.app.root.after(0, self.app.update_training_status, epoch, self.epochs)
                
                def on_epoch_end(self, epoch, logs=None):
                    # Update progress in main thread
                    progress = int(((epoch + 1) / self.epochs) * 100)
                    self.app.root.after(0, self.app.update_training_progress, progress)
            
            progress_callback = ProgressCallback(self, epochs)
            
            # Train model
            history = self.model.train(features_file, epochs=epochs, batch_size=batch_size, 
                                      validation_split=validation_split, 
                                      custom_callbacks=[progress_callback])
            
            # Update UI after training
            if history:
                self.root.after(0, self.training_complete, True)
            else:
                self.root.after(0, self.training_complete, False)
                
        except Exception as e:
            print(f"Error during training: {e}")
            self.root.after(0, self.training_complete, False, str(e))
    
    def update_training_status(self, epoch, total_epochs):
        """Update training status text."""
        self.train_status_var.set(f"Training: Epoch {epoch+1}/{total_epochs}")
    
    def update_training_progress(self, progress):
        """Update training progress bar."""
        self.train_progress["value"] = progress
    
    def training_complete(self, success, error=None):
        """Handle completion of model training."""
        # Re-enable training button
        self.train_btn.config(state="normal")
        
        if success:
            self.train_status_var.set("Training complete")
            self.train_progress["value"] = 100
            messagebox.showinfo("Success", "Model training complete.")
            self.add_activity("Model training completed successfully")
            
            # Update training history visualization
            self.update_training_history()
        else:
            self.train_status_var.set("Training failed")
            error_msg = f"Error: {error}" if error else "Unknown error"
            messagebox.showerror("Error", f"Model training failed. {error_msg}")
            self.add_activity("Model training failed")
    
    def start_recognition(self):
        """Start real-time face recognition."""
        # Check if model exists
        if not os.path.exists(os.path.join("models", "face_recognition_model.keras")):
            messagebox.showinfo("Info", "No trained model available. Please train the model first.")
            return
        
        # Load the model
        try:
            self.model.load_model()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return
        
        # Initialize recognizer
        self.recognizer = FaceRecognizer(self.model, min_confidence=float(self.confidence_var.get()))
        
        # Start camera
        self.camera_id = int(self.rec_camera_var.get())
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        
        # Update UI
        self.start_recognition_btn.config(state="disabled")
        self.stop_recognition_btn.config(state="normal")
        
        # Update status message based on context
        if self.current_attendance_context:
            course_name = self.current_attendance_context['course_name']
            self.status_var.set(f"Recognition running for course: {course_name}")
            self.add_activity(f"Started face recognition for course: {course_name}")
        else:
            self.status_var.set("Recognition running...")
            self.add_activity("Started face recognition")
        
        # Set flag and start thread
        self.recognition_running = True
        
        # Start frame update thread if not already running
        if self.frame_thread is None or not self.frame_thread.is_alive():
            self.frame_thread = threading.Thread(target=self.update_frame)
            self.frame_thread.daemon = True
            self.frame_thread.start()
    
    def stop_recognition(self):
        """Stop real-time face recognition."""
        self.recognition_running = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Update UI        self.start_recognition_btn.config(state="normal")
        self.stop_recognition_btn.config(state="disabled")
        self.status_var.set("Recognition stopped")
        
        self.add_activity("Stopped face recognition")
    def update_attendance(self, person_name):
        """
        Update attendance records when a person is recognized.
        
        This method handles attendance tracking with course context awareness:
        1. Checks if there's an active course context
        2. Verifies if the person is enrolled in the course
        3. Updates existing attendance records or creates new ones
        4. Calculates duration between check-ins
        5. Updates the attendance display
        
        Args:
            person_name (str): The name of the recognized person
            
        Returns:
            None
        """
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        course_name = "General"
        course_id = None
        
        # If we have an active course context, use it
        if self.current_attendance_context:
            course_id = self.current_attendance_context['course_id']
            course_name = self.current_attendance_context['course_name']
            
            # Check if person is enrolled in this course
            if course_id in self.courses:
                students = self.courses[course_id].get('students', [])
                if person_name not in students:
                    # Skip attendance for students not in the course
                    message = f"{person_name} detected but not enrolled in course {course_name}"
                    self.logger.info(message)
                    self.status_var.set(message)
                    return
        
        # Initialize person's attendance records if not exists
        if person_name not in self.attendance_records:
            self.attendance_records[person_name] = []
        
        # Check if we already have an open attendance record for today
        today = current_time.split(' ')[0]  # Get date part
        
        # Look for an existing record for the same person and course today
        existing_record = None
        for record in self.attendance_records[person_name]:
            record_date = record['timestamp'].split(' ')[0]
            if record_date == today and record.get('course_id') == course_id:
                existing_record = record
                break
        
        if existing_record:
            # Update existing record
            existing_record['time_out'] = current_time
            
            # Calculate duration
            try:
                time_in_obj = time.strptime(existing_record['time_in'], "%Y-%m-%d %H:%M:%S")
                current_time_obj = time.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                
                duration_seconds = time.mktime(current_time_obj) - time.mktime(time_in_obj)
                hours, remainder = divmod(duration_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                duration_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                
                existing_record['duration'] = duration_str
                
                # Update UI
                self.update_attendance_display()
                
            except Exception as e:
                print(f"Error calculating duration: {e}")
        else:
            # Create new record
            record_id = str(time.time())  # Unique ID based on timestamp
            new_record = {
                'id': record_id,
                'timestamp': current_time,
                'time_in': current_time,
                'time_out': current_time,
                'duration': "00:00:00",
                'course_id': course_id,
                'course_name': course_name
            }
            
            self.attendance_records[person_name].append(new_record)
            
            # Update UI
            self.update_attendance_display()
            
            self.add_activity(f"{person_name} detected - attendance recorded for {course_name}")
    
    def update_attendance_display(self):
        """Update the attendance treeview display."""
        # Clear existing items
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        # Get filters
        selected_course = self.attendance_course_var.get()
        selected_date = self.attendance_date_var.get()
        
        # Add records to treeview
        for person_name, records in self.attendance_records.items():
            for record in records:
                if not isinstance(record, dict):
                    continue  # Skip legacy format records
                
                # Extract date from timestamp
                record_date = record['timestamp'].split(' ')[0]
                record_time = record['timestamp'].split(' ')[1]
                
                course_name = record.get('course_name', 'General')
                
                # Apply filters
                if selected_course != "All Courses" and course_name != selected_course:
                    continue
                    
                if selected_date != "All Dates" and record_date != selected_date:
                    continue
                
                # Format values
                time_in = record.get('time_in', record['timestamp'])
                time_out = record.get('time_out', record['timestamp'])
                duration = record.get('duration', "00:00:00")
                
                # Add to treeview
                self.attendance_tree.insert('', 'end', values=(
                    person_name,
                    course_name,
                    record_date,
                    time_in.split(' ')[1],  # Just show time part
                    time_out.split(' ')[1],  # Just show time part
                    duration
                ))
        
        # Update filters        self.update_attendance_date_filter()
    
    def export_attendance(self):
        """Export attendance records to a CSV file."""
        if not self.attendance_tree.get_children():
            messagebox.showinfo("Info", "No attendance records to export")
            return
        
        # Ask for file to save
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Attendance Records"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            import csv
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(["Name", "Course", "Date", "Time In", "Time Out", "Duration"])
                
                # Write data from tree
                for item_id in self.attendance_tree.get_children():
                    values = self.attendance_tree.item(item_id, 'values')
                    writer.writerow(values)
                
            messagebox.showinfo("Success", f"Attendance records exported to {file_path}")
            self.add_activity(f"Exported attendance records to CSV file")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def clear_attendance(self):
        """Clear all attendance records."""
        if not self.attendance_tree.get_children():
            return
            
        if messagebox.askyesno("Confirm", "Clear all attendance records?"):
            # Clear treeview
            for item in self.attendance_tree.get_children():
                self.attendance_tree.delete(item)
            
            # Clear records dict
            self.attendance_records.clear()
            
            self.add_activity("Cleared all attendance records")
    
    def add_activity(self, message):
        """Add an activity message to the dashboard."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.activity_text.config(state="normal")
        self.activity_text.insert(tk.END, log_message)
        self.activity_text.see(tk.END)
        self.activity_text.config(state="disabled")
        
        # Log to file as well
        self.logger.info(message)
    
    def on_close(self):
        """Handle window close event."""
        # Stop any running processes
        if self.is_capturing or self.recognition_running:
            if self.cap is not None:
                self.cap.release()
            
            self.is_capturing = False
            self.recognition_running = False
            
            # Wait for threads to finish
            if self.frame_thread is not None and self.frame_thread.is_alive():
                self.frame_thread.join(timeout=1.0)
        
        # Close the window
        self.root.destroy()

    def upload_photo(self):
        """Upload a photo for face recognition."""
        file_path = filedialog.askopenfilename(
            title="Select Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
            
        self.current_photo_path = file_path
        self.processed_photo = None
        self.photo_recognition_results = []
        
        # Display original image
        try:
            # Read image with PIL for display
            pil_img = Image.open(file_path)
            
            # Resize while maintaining aspect ratio
            img_width, img_height = pil_img.size
            canvas_width = self.photo_canvas.winfo_width()
            canvas_height = self.photo_canvas.winfo_height()
            
            # Calculate new dimensions
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_tk = ImageTk.PhotoImage(pil_img)
            
            # Show on canvas
            self.photo_canvas.create_image(
                canvas_width//2, canvas_height//2,
                image=self.photo_tk, anchor=tk.CENTER
            )
            
            # Update status
            self.photo_results_text.delete(1.0, tk.END)
            self.photo_results_text.insert(tk.END, f"Photo loaded: {os.path.basename(file_path)}\n")
            self.photo_results_text.insert(tk.END, "Click 'Process Photo' to detect and recognize faces.")
            
            # Enable process button
            self.process_photo_btn.config(state="normal")
            self.save_result_btn.config(state="disabled")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.photo_results_text.delete(1.0, tk.END)
            self.photo_results_text.insert(tk.END, f"Error loading image: {str(e)}")
    
    def process_photo(self):
        """Process the uploaded photo and recognize faces."""
        if not self.current_photo_path:
            messagebox.showinfo("Info", "Please upload a photo first.")
            return
            
        try:
            # Ensure the model is loaded
            if not hasattr(self.model, 'model') or self.model.model is None:
                model_loaded = self.model.load_model()
                if not model_loaded:
                    messagebox.showerror("Error", "Failed to load the recognition model.")
                    return
            
            # Set confidence threshold
            min_confidence = self.photo_confidence_var.get()
            
            # Initialize recognizer if needed
            if self.recognizer is None:
                self.recognizer = FaceRecognizer(self.model, min_confidence=min_confidence)
            else:
                self.recognizer.min_confidence = min_confidence
            
            # Recognize faces in the image
            processed_img, recognition_results = self.recognizer.recognize_image(self.current_photo_path)
            
            if processed_img is None:
                messagebox.showerror("Error", "Failed to process the image.")
                return
                
            self.processed_photo = processed_img
            self.photo_recognition_results = recognition_results
            
            # Convert OpenCV image to PIL format
            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(processed_img_rgb)
            
            # Resize while maintaining aspect ratio
            img_width, img_height = pil_img.size
            canvas_width = self.photo_canvas.winfo_width()
            canvas_height = self.photo_canvas.winfo_height()
            
            # Calculate new dimensions
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_tk = ImageTk.PhotoImage(pil_img)
            
            # Show on canvas
            self.photo_canvas.delete("all")
            self.photo_canvas.create_image(
                canvas_width//2, canvas_height//2,
                image=self.photo_tk, anchor=tk.CENTER
            )
            
            # Update results text
            self.photo_results_text.delete(1.0, tk.END)
            if recognition_results:
                self.photo_results_text.insert(tk.END, f"Found {len(recognition_results)} face(s):\n\n")
                for i, (_, _, _, _, label, confidence) in enumerate(recognition_results):
                    self.photo_results_text.insert(tk.END, f"Face {i+1}: {label} ({confidence:.2f})\n")
                
                # Enable save button
                self.save_result_btn.config(state="normal")
            else:
                self.photo_results_text.insert(tk.END, "No faces detected in the image.")
                self.save_result_btn.config(state="disabled")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error processing photo: {str(e)}")
            self.photo_results_text.delete(1.0, tk.END)
            self.photo_results_text.insert(tk.END, f"Error processing photo: {str(e)}")
    
    def save_photo_result(self):
        """Save the processed photo with face recognition results."""
        if self.processed_photo is None:
            messagebox.showinfo("Info", "No processed photo to save.")
            return
            
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")],
            title="Save Processed Photo"
        )
        
        if not save_path:
            return
            
        try:
            cv2.imwrite(save_path, self.processed_photo)
            messagebox.showinfo("Success", f"Processed photo saved to {save_path}")
            
            # Also update attendance records if there are recognized persons
            for _, _, _, _, label, confidence in self.photo_recognition_results:
                if label != "Unknown" and confidence >= self.photo_confidence_var.get():
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    if label not in self.attendance_records:
                        self.attendance_records[label] = []
                        
                    self.attendance_records[label].append({
                        'timestamp': timestamp,
                        'source': 'photo',
                        'confidence': confidence
                    })
                    
                    self.logger.info(f"{label} detected in photo - attendance recorded")
            
            # Refresh attendance tab if needed
            self.update_attendance_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save photo: {str(e)}")

# ---- Course Management Methods ----
    
    def load_courses(self):
        """Load saved courses from file."""
        try:
            course_file = os.path.join('data', 'courses.json')
            if os.path.exists(course_file):
                import json
                with open(course_file, 'r') as f:
                    self.courses = json.load(f)
                self.logger.info(f"Loaded {len(self.courses)} courses")
            else:
                # Create sample courses if no courses exist
                self.create_sample_courses()
        except Exception as e:
            self.logger.error(f"Error loading courses: {e}")
            self.courses = {}
    
    def create_sample_courses(self):
        """Create sample courses for demonstration."""
        import uuid
        
        # Sample course 1
        course_id1 = str(uuid.uuid4())
        self.courses[course_id1] = {
            'name': 'Introduction to AI',
            'schedule': '10:00-12:00',
            'days': ['Monday', 'Wednesday'],
            'students': []
        }
        
        # Sample course 2
        course_id2 = str(uuid.uuid4())
        self.courses[course_id2] = {
            'name': 'Computer Vision',
            'schedule': '14:00-16:00',
            'days': ['Tuesday', 'Thursday'],
            'students': []
        }
        
        # Sample course 3
        course_id3 = str(uuid.uuid4())
        self.courses[course_id3] = {
            'name': 'Machine Learning Lab',
            'schedule': '09:00-11:00',
            'days': ['Friday'],
            'students': []
        }
        
        # Save the sample courses
        self.save_courses()
        self.logger.info("Created sample courses")
    def save_courses(self):
        """Save courses to file."""
        try:
            course_file = os.path.join('data', 'courses.json')
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(course_file), exist_ok=True)
            
            import json
            with open(course_file, 'w') as f:
                json.dump(self.courses, f, indent=4)
            self.logger.info(f"Saved {len(self.courses)} courses")
            
            # Update dashboard
            try:
                self.update_dashboard_courses()
            except:
                pass  # In case the dashboard hasn't been initialized yet
            
            # Update attendance course filter
            try:
                self.update_attendance_course_filter()
            except:
                pass  # In case the attendance tab hasn't been initialized yet
                
        except Exception as e:
            self.logger.error(f"Error saving courses: {e}")
            messagebox.showerror("Error", f"Failed to save courses: {e}")
    
    def update_course_list(self):
        """Update the course list in the treeview."""
        # Clear existing items
        for item in self.course_tree.get_children():
            self.course_tree.delete(item)
        
        # Add courses
        for course_id, course_info in self.courses.items():
            days_str = ", ".join(course_info.get('days', []))
            self.course_tree.insert('', 'end', values=(
                course_info['name'],
                course_info.get('schedule', ''),
                days_str
            ), tags=(course_id,))
    
    def on_course_selected(self, event):
        """Handle course selection event."""
        selected_items = self.course_tree.selection()
        if not selected_items:
            self.current_course = None
            return
        
        # Get course ID from tags
        course_id = self.course_tree.item(selected_items[0], 'tags')[0]
        self.current_course = course_id
        
        # Update student list
        self.update_student_list()
    
    def update_student_list(self):
        """Update the student list for the selected course."""
        # Clear existing items
        for item in self.student_tree.get_children():
            self.student_tree.delete(item)
        
        if not self.current_course or self.current_course not in self.courses:
            return
        
        # Add students
        students = self.courses[self.current_course].get('students', [])
        for student in students:
            self.student_tree.insert('', 'end', values=(student,))
    
    def add_course_dialog(self):
        """Show dialog to add a new course."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Course")
        dialog.geometry("400x300")
        dialog.grab_set()  # Make it modal
        
        ttk.Label(dialog, text="Course Name:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var, width=30).grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="Schedule (HH:MM-HH:MM):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        schedule_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=schedule_var, width=30).grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="Days:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        
        # Checkboxes for days
        days_frame = ttk.Frame(dialog)
        days_frame.grid(row=2, column=1, sticky="w", padx=10, pady=5)
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_vars = []
        
        for i, day in enumerate(days):
            var = tk.BooleanVar()
            day_vars.append(var)
            ttk.Checkbutton(days_frame, text=day, variable=var).grid(row=i//3, column=i%3, sticky="w", padx=5, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=3, column=0, columnspan=2, pady=15)
        
        ttk.Button(button_frame, text="Save", command=lambda: self.save_new_course(
            name_var.get(), schedule_var.get(), [days[i] for i, var in enumerate(day_vars) if var.get()], dialog
        )).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    
    def save_new_course(self, name, schedule, days, dialog):
        """Save a new course."""
        if not name:
            messagebox.showerror("Error", "Course name is required")
            return
        
        if not schedule:
            messagebox.showerror("Error", "Schedule is required")
            return
        
        if not days:
            messagebox.showerror("Error", "At least one day must be selected")
            return
        
        # Validate schedule format (HH:MM-HH:MM)
        import re
        if not re.match(r'^\d{1,2}:\d{2}-\d{1,2}:\d{2}$', schedule):
            messagebox.showerror("Error", "Schedule must be in format HH:MM-HH:MM")
            return
        
        # Create course ID
        import uuid
        course_id = str(uuid.uuid4())
        
        # Add course
        self.courses[course_id] = {
            'name': name,
            'schedule': schedule,
            'days': days,
            'students': []
        }
        
        # Save and update
        self.save_courses()
        self.update_course_list()
        
        # Close dialog
        dialog.destroy()
    
    def edit_course_dialog(self):
        """Show dialog to edit a course."""
        if not self.current_course:
            messagebox.showinfo("Info", "Please select a course to edit")
            return
        
        course = self.courses[self.current_course]
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Course")
        dialog.geometry("400x300")
        dialog.grab_set()  # Make it modal
        
        ttk.Label(dialog, text="Course Name:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        name_var = tk.StringVar(value=course['name'])
        ttk.Entry(dialog, textvariable=name_var, width=30).grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="Schedule (HH:MM-HH:MM):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        schedule_var = tk.StringVar(value=course.get('schedule', ''))
        ttk.Entry(dialog, textvariable=schedule_var, width=30).grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(dialog, text="Days:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        
        # Checkboxes for days
        days_frame = ttk.Frame(dialog)
        days_frame.grid(row=2, column=1, sticky="w", padx=10, pady=5)
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_vars = []
        
        course_days = course.get('days', [])
        
        for i, day in enumerate(days):
            var = tk.BooleanVar(value=day in course_days)
            day_vars.append(var)
            ttk.Checkbutton(days_frame, text=day, variable=var).grid(row=i//3, column=i%3, sticky="w", padx=5, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=3, column=0, columnspan=2, pady=15)
        
        ttk.Button(button_frame, text="Save", command=lambda: self.save_edited_course(
            name_var.get(), schedule_var.get(), [days[i] for i, var in enumerate(day_vars) if var.get()], dialog
        )).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    
    def save_edited_course(self, name, schedule, days, dialog):
        """Save an edited course."""
        if not name:
            messagebox.showerror("Error", "Course name is required")
            return
        
        if not schedule:
            messagebox.showerror("Error", "Schedule is required")
            return
        
        if not days:
            messagebox.showerror("Error", "At least one day must be selected")
            return
        
        # Validate schedule format (HH:MM-HH:MM)
        import re
        if not re.match(r'^\d{1,2}:\d{2}-\d{1,2}:\d{2}$', schedule):
            messagebox.showerror("Error", "Schedule must be in format HH:MM-HH:MM")
            return
        
        # Get existing students
        existing_students = self.courses[self.current_course].get('students', [])
        
        # Update course
        self.courses[self.current_course].update({
            'name': name,
            'schedule': schedule,
            'days': days,
            'students': existing_students
        })
        
        # Save and update
        self.save_courses()
        self.update_course_list()
        
        # Close dialog
        dialog.destroy()
    
    def delete_course(self):
        """Delete the selected course."""
        if not self.current_course:
            messagebox.showinfo("Info", "Please select a course to delete")
            return
        
        # Confirm deletion
        course_name = self.courses[self.current_course]['name']
        if not messagebox.askyesno("Confirm", f"Are you sure you want to delete the course '{course_name}'?"):
            return
        
        # Delete course
        del self.courses[self.current_course]
        
        # Save and update
        self.save_courses()
        self.update_course_list()
        
        # Clear current course
        self.current_course = None
        self.update_student_list()
    
    def add_student_to_course(self):
        """Add a student to the selected course."""
        if not self.current_course:
            messagebox.showinfo("Info", "Please select a course first")
            return
        
        # Get available faces from dataset
        available_faces = self.get_available_faces()
        
        if not available_faces:
            messagebox.showinfo("Info", "No faces available in the dataset")
            return
        
        # Get current students in course
        current_students = self.courses[self.current_course].get('students', [])
        
        # Show selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Student to Course")
        dialog.geometry("400x300")
        dialog.grab_set()  # Make it modal
        
        ttk.Label(dialog, text="Select student to add:").pack(anchor="w", padx=10, pady=5)
        
        # Create listbox with available faces
        listbox_frame = ttk.Frame(dialog)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        face_listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set)
        scrollbar.config(command=face_listbox.yview)
        face_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Add faces to listbox (exclude ones already in the course)
        available_students = [face for face in available_faces if face not in current_students]
        
        if not available_students:
            messagebox.showinfo("Info", "All available faces are already added to this course")
            dialog.destroy()
            return
        
        for face in available_students:
            face_listbox.insert(tk.END, face)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Add", command=lambda: self.add_selected_student(
            face_listbox.get(face_listbox.curselection()[0]) if face_listbox.curselection() else None, dialog
        )).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    
    def add_selected_student(self, student, dialog):
        """Add selected student to the course."""
        if not student:
            messagebox.showinfo("Info", "Please select a student")
            return
        
        # Add student to course
        self.courses[self.current_course].setdefault('students', []).append(student)
        
        # Save and update
        self.save_courses()
        self.update_student_list()
        
        # Close dialog
        dialog.destroy()
    
    def remove_student_from_course(self):
        """Remove a student from the selected course."""
        selected_items = self.student_tree.selection()
        if not selected_items:
            messagebox.showinfo("Info", "Please select a student to remove")
            return
        
        # Get student name
        student_name = self.student_tree.item(selected_items[0], 'values')[0]
        
        # Confirm removal
        if not messagebox.askyesno("Confirm", f"Are you sure you want to remove '{student_name}' from this course?"):
            return
        
        # Remove student
        students = self.courses[self.current_course].get('students', [])
        students.remove(student_name)
        self.courses[self.current_course]['students'] = students
        
        # Save and update
        self.save_courses()
        self.update_student_list()
    
    def get_available_faces(self):
        """Get list of available face names from the dataset."""
        # Check in raw data directory
        raw_dir = os.path.join('data', 'raw')
        if not os.path.exists(raw_dir):
            return []
        
        available_faces = []
        for item in os.listdir(raw_dir):
            item_path = os.path.join(raw_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                available_faces.append(item)
        
        return available_faces
    def take_course_attendance(self):
        """
        Start recognition for taking attendance for the selected course.
        
        This method:
        1. Sets up a course-specific attendance context
        2. Switches to the recognition tab
        3. Starts the face recognition process
        4. Only records attendance for students enrolled in the course
        """
        if not self.current_course:
            messagebox.showinfo("Info", "Please select a course first")
            return
        
        course_name = self.courses[self.current_course]['name']
        students = self.courses[self.current_course].get('students', [])
        
        if not students:
            messagebox.showinfo("Info", "This course has no students")
            return
        
        # Switch to recognition tab and start recognition
        self.select_tab(4)  # Index of Recognition tab
        
        # Set course context for attendance
        self.current_attendance_context = {
            'course_id': self.current_course,
            'course_name': course_name
        }
        
        # Start recognition
        self.start_recognition()

    def update_attendance_course_filter(self):
        """Update the course filter dropdown with available courses."""
        courses = ["All Courses"]
        
        # Add course names
        for course_id, course_info in self.courses.items():
            courses.append(course_info['name'])
        
        self.course_combobox['values'] = courses
        
        # Update dates based on selected course
        self.update_attendance_date_filter()
    
    def update_attendance_date_filter(self):
        """Update the date filter dropdown with available dates."""
        dates = ["All Dates"]
        
        # Get unique dates from attendance records
        for record_list in self.attendance_records.values():
            for record in record_list:
                if isinstance(record, dict) and 'timestamp' in record:
                    date = record['timestamp'].split(' ')[0]  # Extract date part
                    if date not in dates:
                        dates.append(date)
        
        self.date_combobox['values'] = sorted(dates)
    
    def filter_attendance_by_course(self, event=None):
        """Filter attendance records by selected course."""
        self.update_attendance_display()
    
    def filter_attendance_by_date(self, event=None):
        """Filter attendance records by selected date."""
        self.update_attendance_display()

    def update_dashboard_courses(self):
        """Update the course information displayed on the dashboard."""
        # Clear existing items
        for item in self.dashboard_course_tree.get_children():
            self.dashboard_course_tree.delete(item)
        
        # Add courses
        for course_id, course_info in self.courses.items():
            days_str = ", ".join(course_info.get('days', []))
            student_count = len(course_info.get('students', []))
            
            self.dashboard_course_tree.insert('', 'end', values=(
                course_info['name'],
                course_info.get('schedule', ''),
                days_str,
                student_count
            ))

def main():
    """Main function to start the application."""
    root = tk.Tk()
    app = AttendanceSystemApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
