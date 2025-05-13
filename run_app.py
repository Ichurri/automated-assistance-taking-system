#!/usr/bin/env python
"""
Face Recognition Attendance System with Course Management

This script runs the Face Recognition Attendance System application.
The system provides the following features:
- Face capture and dataset management
- Face recognition model training
- Real-time face recognition for attendance tracking
- Course management (registration, schedules, student assignment)
- Course-specific attendance tracking
- Attendance record management and export

Run this script to start the application.
"""

import os
import sys
import tkinter as tk

# Add the parent directory to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import from our gui module
from gui.app import AttendanceSystemApp

def main():
    """Main function to start the application."""
    root = tk.Tk()
    app = AttendanceSystemApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
