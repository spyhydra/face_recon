import os
import platform

# Determine the correct path separator based on the operating system
PATH_SEP = '\\' if platform.system() == 'Windows' else '/'

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(BASE_DIR, "TrainingImage")
STUDENT_DETAILS_DIR = os.path.join(BASE_DIR, "StudentDetails")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
UI_IMAGES_DIR = os.path.join(BASE_DIR, "UI_Image")
TRAINING_IMAGE_LABEL_DIR = os.path.join(BASE_DIR, "TrainingImageLabel")

# Ensure directories exist
for directory in [TRAINING_DIR, STUDENT_DETAILS_DIR, ATTENDANCE_DIR, 
                 TRAINING_IMAGE_LABEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Files
HAARCASCADE_FRONTAL_FACE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
HAARCASCADE_ALT_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_alt.xml")
STUDENT_DETAILS_PATH = os.path.join(STUDENT_DETAILS_DIR, "studentdetails.csv")
TRAINER_PATH = os.path.join(TRAINING_IMAGE_LABEL_DIR, "Trainner.yml")
LOGO_PATH = os.path.join(UI_IMAGES_DIR, "0001.png")
ICON_PATH = os.path.join(BASE_DIR, "AMS.ico")

# Face recognition settings
FACE_DETECTION_CONFIDENCE = 45  # Lower value = stricter matching
FACE_DETECTION_SCALE_FACTOR = 1.2
FACE_DETECTION_MIN_NEIGHBORS = 5
SAMPLE_COUNT = 50  # Number of images to capture per person

# UI settings
UI_THEME = {
    "bg_color": "#1c1c1c",
    "fg_color": "white",
    "button_bg": "#333333",
    "button_fg": "yellow",
    "highlight_bg": "#0078D7",
    "error_fg": "yellow",
    "success_fg": "green",
    "header_bg": "#2c3e50",
}

# CSV column names
STUDENT_COLUMNS = ["Enrollment", "Name"]
ATTENDANCE_COLUMNS = ["Enrollment", "Name", "Date", "Time"] 