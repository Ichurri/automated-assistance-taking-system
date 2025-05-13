import os
import sys
import tkinter as tk
from gui.app import AttendanceSystemApp

def main():
    """Launch the Face Recognition Attendance System GUI."""
    root = tk.Tk()
    app = AttendanceSystemApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
