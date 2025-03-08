import tkinter as tk
from tkinter import *
import os
import cv2
import numpy as np
from PIL import ImageTk, Image
import datetime
import time
import pyttsx3
import threading

# Import our utility modules
import config
import face_utils
import db_utils
import ui_utils

# Import other modules
import show_attendance
import takeImage
import trainImage
import automaticAttedance

# Initialize text-to-speech engine
tts_engine = None

def get_tts_engine():
    """Lazy initialization of text-to-speech engine"""
    global tts_engine
    if tts_engine is None:
        tts_engine = pyttsx3.init()
    return tts_engine

def text_to_speech(user_text):
    """Convert text to speech"""
    try:
        engine = get_tts_engine()
        
        # Run in a separate thread to avoid blocking the UI
        def speak():
            engine.say(user_text)
            engine.runAndWait()
        
        threading.Thread(target=speak, daemon=True).start()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

# Create main window
window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry("1280x720")
window.resizable(True, True)
window.configure(background=config.UI_THEME["bg_color"])

# Set window icon
try:
    window.iconbitmap(config.ICON_PATH)
except:
    pass  # Icon not found, continue without it

# Create a title frame
title_frame = tk.Frame(window, bg=config.UI_THEME["bg_color"], relief=RIDGE, bd=10)
title_frame.pack(fill=X)

# Add logo
try:
    logo = ui_utils.load_and_resize_image(config.LOGO_PATH, 50, 47)
    logo_label = tk.Label(title_frame, image=logo, bg=config.UI_THEME["bg_color"])
    logo_label.image = logo  # Keep a reference
    logo_label.pack(side=LEFT, padx=10)
except:
    pass  # Logo not found, continue without it

# Add title
title_label = tk.Label(
    title_frame, 
    text="Face Recognition Attendance System",
    bg=config.UI_THEME["bg_color"],
    fg=config.UI_THEME["fg_color"],
    font=("Verdana", 30, "bold")
)
title_label.pack(side=LEFT, padx=10)

# Create a frame for the main content
content_frame = tk.Frame(window, bg=config.UI_THEME["bg_color"])
content_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

# Create a frame for student registration
registration_frame = tk.Frame(content_frame, bg=config.UI_THEME["bg_color"], bd=2, relief=RIDGE)
registration_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)

# Add a title for registration frame
registration_title = tk.Label(
    registration_frame,
    text="Student Registration",
    bg=config.UI_THEME["bg_color"],
    fg=config.UI_THEME["fg_color"],
    font=("Verdana", 20, "bold")
)
registration_title.pack(pady=10)

# Create validation for enrollment number (only digits)
def validate_enrollment(text):
    return text.isdigit() or text == ""

validate_cmd = (window.register(validate_enrollment), '%P')

# Create enrollment entry
enrollment_frame, enrollment_entry = ui_utils.create_entry_with_label(
    registration_frame,
    "Enrollment:",
    validate_cmd,
    width=20
)
enrollment_frame.pack(pady=10)

# Create name entry
name_frame, name_entry = ui_utils.create_entry_with_label(
    registration_frame,
    "Name:",
    width=20
)
name_frame.pack(pady=10)

# Create a message label
message_label = tk.Label(
    registration_frame,
    text="",
    bg=config.UI_THEME["bg_color"],
    fg=config.UI_THEME["fg_color"],
    font=("Verdana", 12),
    wraplength=400
)
message_label.pack(pady=10)

# Define functions for buttons
def clear_fields():
    """Clear all input fields"""
    enrollment_entry.delete(0, END)
    name_entry.delete(0, END)
    message_label.config(text="")

def take_image_button_action():
    """Action for Take Image button"""
    enrollment = enrollment_entry.get()
    name = name_entry.get()
    
    if not enrollment or not name:
        ui_utils.show_message(
            window,
            "Error",
            "Please enter both Enrollment Number and Name.",
            "error"
        )
        text_to_speech("Please enter both Enrollment Number and Name.")
        return
    
    # Take images
    result = takeImage.TakeImage(
        enrollment,
        name,
        message_label,
        text_to_speech
    )
    
    if result:
        # Clear fields on success
        clear_fields()

def train_image_button_action():
    """Action for Train Image button"""
    # Train the model
    trainImage.TrainImage(
        message_label,
        text_to_speech,
        window
    )

# Create buttons for registration
button_frame = tk.Frame(registration_frame, bg=config.UI_THEME["bg_color"])
button_frame.pack(pady=20)

take_image_button = ui_utils.create_rounded_button(
    button_frame,
    "Take Images",
    take_image_button_action,
    width=15,
    height=2
)
take_image_button.pack(side=LEFT, padx=10)

train_image_button = ui_utils.create_rounded_button(
    button_frame,
    "Train Model",
    train_image_button_action,
    width=15,
    height=2
)
train_image_button.pack(side=LEFT, padx=10)

clear_button = ui_utils.create_rounded_button(
    button_frame,
    "Clear",
    clear_fields,
    width=15,
    height=2
)
clear_button.pack(side=LEFT, padx=10)

# Create a frame for attendance
attendance_frame = tk.Frame(content_frame, bg=config.UI_THEME["bg_color"], bd=2, relief=RIDGE)
attendance_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)

# Add a title for attendance frame
attendance_title = tk.Label(
    attendance_frame,
    text="Attendance Management",
    bg=config.UI_THEME["bg_color"],
    fg=config.UI_THEME["fg_color"],
    font=("Verdana", 20, "bold")
)
attendance_title.pack(pady=10)

# Create subject entry
subject_frame, subject_entry = ui_utils.create_entry_with_label(
    attendance_frame,
    "Subject:",
    width=20
)
subject_frame.pack(pady=10)

# Create a message label for attendance
attendance_message = tk.Label(
    attendance_frame,
    text="",
    bg=config.UI_THEME["bg_color"],
    fg=config.UI_THEME["fg_color"],
    font=("Verdana", 12),
    wraplength=400
)
attendance_message.pack(pady=10)

# Define functions for attendance buttons
def automatic_attendance_button_action():
    """Action for Automatic Attendance button"""
    subject = subject_entry.get()
    
    if not subject:
        ui_utils.show_message(
            window,
            "Error",
            "Please enter a subject name.",
            "error"
        )
        text_to_speech("Please enter a subject name.")
        return
    
    # Take automatic attendance
    automaticAttedance.take_attendance(
        subject,
        attendance_message,
        text_to_speech,
        window
    )

def view_attendance_button_action():
    """Action for View Attendance button"""
    subject = subject_entry.get()
    
    if not subject:
        ui_utils.show_message(
            window,
            "Error",
            "Please enter a subject name.",
            "error"
        )
        text_to_speech("Please enter a subject name.")
        return
    
    # Show attendance
    show_attendance.show_attendance(subject, window)

# Create buttons for attendance
attendance_button_frame = tk.Frame(attendance_frame, bg=config.UI_THEME["bg_color"])
attendance_button_frame.pack(pady=20)

automatic_attendance_button = ui_utils.create_rounded_button(
    attendance_button_frame,
    "Take Attendance",
    automatic_attendance_button_action,
    width=15,
    height=2
)
automatic_attendance_button.pack(side=LEFT, padx=10)

view_attendance_button = ui_utils.create_rounded_button(
    attendance_button_frame,
    "View Attendance",
    view_attendance_button_action,
    width=15,
    height=2
)
view_attendance_button.pack(side=LEFT, padx=10)

# Create a status bar
status_bar = tk.Label(
    window,
    text="Ready",
    bd=1,
    relief=SUNKEN,
    anchor=W,
    bg=config.UI_THEME["bg_color"],
    fg=config.UI_THEME["fg_color"]
)
status_bar.pack(side=BOTTOM, fill=X)

# Welcome message
text_to_speech("Welcome to Face Recognition Attendance System")

# Start the main loop
if __name__ == "__main__":
    window.mainloop()
