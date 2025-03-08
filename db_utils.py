import os
import csv
import pandas as pd
import datetime
import config

def ensure_csv_exists(file_path, columns):
    """
    Ensure that a CSV file exists with the specified columns
    
    Args:
        file_path: Path to the CSV file
        columns: List of column names
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)

def add_student(enrollment, name):
    """
    Add a new student to the database
    
    Args:
        enrollment: Student enrollment number
        name: Student name
        
    Returns:
        True if successful, False if student already exists
    """
    # Ensure the CSV file exists
    ensure_csv_exists(config.STUDENT_DETAILS_PATH, config.STUDENT_COLUMNS)
    
    # Check if student already exists
    if student_exists(enrollment):
        return False
    
    # Add student to CSV
    with open(config.STUDENT_DETAILS_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([enrollment, name])
    
    return True

def student_exists(enrollment):
    """
    Check if a student exists in the database
    
    Args:
        enrollment: Student enrollment number
        
    Returns:
        True if student exists, False otherwise
    """
    # Ensure the CSV file exists
    ensure_csv_exists(config.STUDENT_DETAILS_PATH, config.STUDENT_COLUMNS)
    
    # Read the CSV file
    try:
        df = pd.read_csv(config.STUDENT_DETAILS_PATH)
        return enrollment in df['Enrollment'].values
    except:
        return False

def get_student_name(enrollment):
    """
    Get a student's name from their enrollment number
    
    Args:
        enrollment: Student enrollment number
        
    Returns:
        Student name if found, None otherwise
    """
    # Ensure the CSV file exists
    ensure_csv_exists(config.STUDENT_DETAILS_PATH, config.STUDENT_COLUMNS)
    
    # Read the CSV file
    try:
        df = pd.read_csv(config.STUDENT_DETAILS_PATH)
        student = df[df['Enrollment'] == enrollment]
        if not student.empty:
            return student['Name'].values[0]
        return None
    except:
        return None

def mark_attendance(enrollment, subject):
    """
    Mark attendance for a student
    
    Args:
        enrollment: Student enrollment number
        subject: Subject name
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get student name
        name = get_student_name(enrollment)
        if name is None:
            return False
        
        # Get current date and time
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Create attendance file path
        attendance_file = os.path.join(
            config.ATTENDANCE_DIR,
            f"{subject}_{date}.csv"
        )
        
        # Ensure the attendance file exists
        ensure_csv_exists(attendance_file, config.ATTENDANCE_COLUMNS)
        
        # Check if attendance already marked
        df = pd.read_csv(attendance_file)
        if enrollment in df['Enrollment'].values:
            return True  # Already marked
        
        # Mark attendance
        with open(attendance_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([enrollment, name, date, time_str])
        
        return True
    except Exception as e:
        print(f"Error marking attendance: {e}")
        return False

def get_attendance(subject, date=None):
    """
    Get attendance for a subject on a specific date
    
    Args:
        subject: Subject name
        date: Date in YYYY-MM-DD format (default: today)
        
    Returns:
        DataFrame with attendance data
    """
    try:
        # Get date
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Create attendance file path
        attendance_file = os.path.join(
            config.ATTENDANCE_DIR,
            f"{subject}_{date}.csv"
        )
        
        # Check if file exists
        if not os.path.isfile(attendance_file):
            return pd.DataFrame(columns=config.ATTENDANCE_COLUMNS)
        
        # Read attendance data
        df = pd.read_csv(attendance_file)
        return df
    except Exception as e:
        print(f"Error getting attendance: {e}")
        return pd.DataFrame(columns=config.ATTENDANCE_COLUMNS)

def get_all_students():
    """
    Get all students in the database
    
    Returns:
        DataFrame with student data
    """
    # Ensure the CSV file exists
    ensure_csv_exists(config.STUDENT_DETAILS_PATH, config.STUDENT_COLUMNS)
    
    # Read the CSV file
    try:
        return pd.read_csv(config.STUDENT_DETAILS_PATH)
    except:
        return pd.DataFrame(columns=config.STUDENT_COLUMNS) 