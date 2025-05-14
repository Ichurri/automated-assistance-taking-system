# Documentation for Course Management Feature

## Overview

The course management feature integrates with the face recognition attendance system, allowing tracking of attendance for specific courses and students. This document outlines the implementation details for developers.

## Key Classes and Methods

### AttendanceSystemApp Class

The main application class handles all aspects of the course management system.

#### Data Structure

Courses are stored in a dictionary with course IDs as keys:

```python
self.courses = {
    "course_id1": {
        "name": "Course Name",
        "schedule": "10:00-12:00",
        "days": ["Monday", "Wednesday"],
        "students": ["Student1", "Student2"]
    },
    "course_id2": { ... }
}
```

#### Course Management Methods

- `load_courses()`: Loads course data from JSON file or creates sample courses
- `save_courses()`: Persists course data to JSON file
- `create_sample_courses()`: Creates demo courses for testing
- `update_course_list()`: Updates course treeview in the UI
- `update_student_list()`: Shows students enrolled in the selected course
- `on_course_selected()`: Handles course selection events
- `add_course_dialog()`: Shows dialog to add a new course
- `edit_course_dialog()`: Shows dialog to edit an existing course
- `delete_course()`: Removes a course from the system
- `add_student_to_course()`: Assigns students from face dataset to courses
- `remove_student_from_course()`: Removes student assignments from courses
- `get_available_faces()`: Gets list of faces from the dataset

#### Attendance Context Methods

- `take_course_attendance()`: Sets up recognition for a specific course
- `update_attendance()`: Records attendance considering course context
- `update_attendance_display()`: Updates the attendance treeview with records
- `update_attendance_course_filter()`: Updates course dropdown in attendance tab
- `update_attendance_date_filter()`: Updates date filter in attendance tab
- `filter_attendance_by_course()`: Filters attendance by course
- `filter_attendance_by_date()`: Filters attendance by date

#### Dashboard Methods

- `update_dashboard_courses()`: Updates course summary on dashboard

#### Photo Attendance Methods

- `update_photo_course_combo()`: Updates course dropdown in photo recognition tab
- `take_photo_attendance()`: Records attendance for a course using processed photo
- `photo_course_attendance()`: Initiates photo attendance for the selected course

## Integration Points

1. **Recognition Flow**:
   - When taking course attendance, the system sets `current_attendance_context`
   - The recognition process checks this context before recording attendance
   - Only students enrolled in the course are marked present

2. **Display Integration**:
   - Course name is displayed on recognition video
   - Non-enrolled students are highlighted in orange
   - Status messages indicate when students aren't enrolled

3. **Attendance Records**:
   - Each attendance record includes:
     - Person name
     - Course ID and name
     - Timestamp, time in, time out
     - Duration
     - Source (camera/photo)

4. **Photo Recognition Integration**:
   - Photos can be used for course-specific attendance
   - System validates recognized faces against course enrollment
   - User interface provides feedback on recognized faces that aren't enrolled
   - Attendance records include "photo" as the source

## File Structure

- `data/courses.json`: Course data persistence
- `gui/app.py`: Course management implementation
    - `setup_course_tab()`: Course UI initialization
    - `setup_attendance_tab()`: Attendance tab with course filtering
    - `setup_photo_recognition_tab()`: Photo recognition with course integration

## Enhancement Ideas

1. **Course Schedule Integration**:
   - Auto-detect current course based on day/time
   - Show upcoming courses on dashboard
   - Send reminders for course attendance

2. **Reporting**:
   - Generate course-specific attendance reports
   - Visualize attendance statistics by course
   - Identify students with low attendance

3. **User Management**:
   - Different roles for instructors and administrators
   - Course-specific permissions
