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

class AttendanceSystem:
    """Main application class for the attendance system"""
    
    def __init__(self, root):
        """Initialize the application"""
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1280x720")
        self.root.resizable(True, True)
        self.root.configure(background=config.UI_THEME["bg_color"])
        
        # Set window icon
        try:
            self.root.iconbitmap(config.ICON_PATH)
        except:
            pass  # Icon not found, continue without it
        
        # Create UI components
        self.create_title_frame()
        self.create_content_frame()
        self.create_registration_frame()
        self.create_attendance_frame()
        self.create_status_bar()
        
        # Welcome message
        text_to_speech("Welcome to Face Recognition Attendance System")
    
    def create_title_frame(self):
        """Create the title frame"""
        self.title_frame = tk.Frame(
            self.root, 
            bg=config.UI_THEME["bg_color"], 
            relief=RIDGE, 
            bd=10
        )
        self.title_frame.pack(fill=X)
        
        # Add logo
        try:
            self.logo = ui_utils.load_and_resize_image(config.LOGO_PATH, 50, 47)
            logo_label = tk.Label(
                self.title_frame, 
                image=self.logo, 
                bg=config.UI_THEME["bg_color"]
            )
            logo_label.pack(side=LEFT, padx=10)
        except:
            pass  # Logo not found, continue without it
        
        # Add title
        title_label = tk.Label(
            self.title_frame, 
            text="Face Recognition Attendance System",
            bg=config.UI_THEME["bg_color"],
            fg=config.UI_THEME["fg_color"],
            font=("Verdana", 30, "bold")
        )
        title_label.pack(side=LEFT, padx=10)
    
    def create_content_frame(self):
        """Create the main content frame"""
        self.content_frame = tk.Frame(
            self.root, 
            bg=config.UI_THEME["bg_color"]
        )
        self.content_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
    
    def create_registration_frame(self):
        """Create the registration frame"""
        self.registration_frame = tk.Frame(
            self.content_frame, 
            bg=config.UI_THEME["bg_color"], 
            bd=2, 
            relief=RIDGE
        )
        self.registration_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)
        
        # Add a title for registration frame
        registration_title = tk.Label(
            self.registration_frame,
            text="Student Registration",
            bg=config.UI_THEME["bg_color"],
            fg=config.UI_THEME["fg_color"],
            font=("Verdana", 20, "bold")
        )
        registration_title.pack(pady=10)
        
        # Create validation for enrollment number (only digits)
        validate_cmd = (self.root.register(self.validate_enrollment), '%P')
        
        # Create enrollment entry
        enrollment_frame, self.enrollment_entry = ui_utils.create_entry_with_label(
            self.registration_frame,
            "Enrollment:",
            validate_cmd,
            width=20
        )
        enrollment_frame.pack(pady=10)
        
        # Create name entry
        name_frame, self.name_entry = ui_utils.create_entry_with_label(
            self.registration_frame,
            "Name:",
            width=20
        )
        name_frame.pack(pady=10)
        
        # Create a message label
        self.message_label = tk.Label(
            self.registration_frame,
            text="",
            bg=config.UI_THEME["bg_color"],
            fg=config.UI_THEME["fg_color"],
            font=("Verdana", 12),
            wraplength=400
        )
        self.message_label.pack(pady=10)
        
        # Create buttons for registration
        button_frame = tk.Frame(self.registration_frame, bg=config.UI_THEME["bg_color"])
        button_frame.pack(pady=20)
        
        take_image_button = ui_utils.create_rounded_button(
            button_frame,
            "Take Images",
            self.take_image_button_action,
            width=15,
            height=2
        )
        take_image_button.pack(side=LEFT, padx=10)
        
        train_image_button = ui_utils.create_rounded_button(
            button_frame,
            "Train Model",
            self.train_image_button_action,
            width=15,
            height=2
        )
        train_image_button.pack(side=LEFT, padx=10)
        
        clear_button = ui_utils.create_rounded_button(
            button_frame,
            "Clear",
            self.clear_fields,
            width=15,
            height=2
        )
        clear_button.pack(side=LEFT, padx=10)
    
    def create_attendance_frame(self):
        """Create the attendance frame"""
        self.attendance_frame = tk.Frame(
            self.content_frame, 
            bg=config.UI_THEME["bg_color"], 
            bd=2, 
            relief=RIDGE
        )
        self.attendance_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)
        
        # Add a title for attendance frame
        attendance_title = tk.Label(
            self.attendance_frame,
            text="Attendance Management",
            bg=config.UI_THEME["bg_color"],
            fg=config.UI_THEME["fg_color"],
            font=("Verdana", 20, "bold")
        )
        attendance_title.pack(pady=10)
        
        # Create subject entry
        subject_frame, self.subject_entry = ui_utils.create_entry_with_label(
            self.attendance_frame,
            "Subject:",
            width=20
        )
        subject_frame.pack(pady=10)
        
        # Create a message label for attendance
        self.attendance_message = tk.Label(
            self.attendance_frame,
            text="",
            bg=config.UI_THEME["bg_color"],
            fg=config.UI_THEME["fg_color"],
            font=("Verdana", 12),
            wraplength=400
        )
        self.attendance_message.pack(pady=10)
        
        # Create buttons for attendance
        attendance_button_frame = tk.Frame(self.attendance_frame, bg=config.UI_THEME["bg_color"])
        attendance_button_frame.pack(pady=20)
        
        automatic_attendance_button = ui_utils.create_rounded_button(
            attendance_button_frame,
            "Take Attendance",
            self.automatic_attendance_button_action,
            width=15,
            height=2
        )
        automatic_attendance_button.pack(side=LEFT, padx=10)
        
        view_attendance_button = ui_utils.create_rounded_button(
            attendance_button_frame,
            "View Attendance",
            self.view_attendance_button_action,
            width=15,
            height=2
        )
        view_attendance_button.pack(side=LEFT, padx=10)
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=SUNKEN,
            anchor=W,
            bg=config.UI_THEME["bg_color"],
            fg=config.UI_THEME["fg_color"]
        )
        self.status_bar.pack(side=BOTTOM, fill=X)
    
    def validate_enrollment(self, text):
        """Validate enrollment number (only digits)"""
        return text.isdigit() or text == ""
    
    def clear_fields(self):
        """Clear all input fields"""
        self.enrollment_entry.delete(0, END)
        self.name_entry.delete(0, END)
        self.message_label.config(text="")
    
    def take_image_button_action(self):
        """Action for Take Image button"""
        enrollment = self.enrollment_entry.get()
        name = self.name_entry.get()
        
        if not enrollment or not name:
            ui_utils.show_message(
                self.root,
                "Error",
                "Please enter both Enrollment Number and Name.",
                "error"
            )
            text_to_speech("Please enter both Enrollment Number and Name.")
            return
        
        # Take images
        import takeImage
        result = takeImage.TakeImage(
            enrollment,
            name,
            self.message_label,
            text_to_speech
        )
        
        if result:
            # Clear fields on success
            self.clear_fields()
    
    def train_image_button_action(self):
        """Action for Train Image button"""
        # Train the model
        import trainImage
        trainImage.TrainImage(
            self.message_label,
            text_to_speech,
            self.root
        )
    
    def automatic_attendance_button_action(self):
        """Action for Automatic Attendance button"""
        subject = self.subject_entry.get()
        
        if not subject:
            ui_utils.show_message(
                self.root,
                "Error",
                "Please enter a subject name.",
                "error"
            )
            text_to_speech("Please enter a subject name.")
            return
        
        # Take automatic attendance
        self.take_attendance(subject)
    
    def take_attendance(self, subject):
        """Take attendance using face recognition"""
        # Check if model exists
        if not os.path.exists(config.TRAINER_PATH):
            ui_utils.show_message(
                self.root,
                "Error",
                "Model not found. Please train the model first.",
                "error"
            )
            text_to_speech("Model not found. Please train the model first.")
            return
        
        # Update status
        self.status_bar.config(text="Taking attendance...")
        self.attendance_message.config(text="Starting camera...")
        
        # Define the attendance task
        def attendance_task():
            try:
                # Initialize camera
                cam = cv2.VideoCapture(0)
                if not cam.isOpened():
                    self.root.after(0, lambda: self.attendance_message.config(
                        text="Could not open camera. Please check your camera connection."
                    ))
                    text_to_speech("Could not open camera. Please check your camera connection.")
                    return
                
                # Create a window for displaying the camera feed
                cv2.namedWindow("Attendance", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Attendance", 640, 480)
                
                # Load student data
                students_df = db_utils.get_all_students()
                if students_df.empty:
                    self.root.after(0, lambda: self.attendance_message.config(
                        text="No students found. Please register students first."
                    ))
                    text_to_speech("No students found. Please register students first.")
                    cam.release()
                    cv2.destroyAllWindows()
                    return
                
                # Set a time limit for attendance (20 seconds)
                start_time = time.time()
                end_time = start_time + 20
                
                # Track recognized students
                recognized_students = set()
                
                # Start capturing frames
                while time.time() < end_time:
                    ret, frame = cam.read()
                    if not ret:
                        break
                    
                    # Display time remaining
                    time_remaining = int(end_time - time.time())
                    frame_display = frame.copy()
                    cv2.putText(
                        frame_display,
                        f"Time remaining: {time_remaining}s",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    # Detect faces
                    try:
                        faces = face_utils.detect_faces(frame)
                    except Exception as e:
                        print(f"MTCNN error: {e}, falling back to Haar Cascade")
                        faces = face_utils.fallback_detect_faces(frame)
                    
                    # Process detected faces
                    for (x, y, w, h) in faces:
                        # Extract face
                        face_img = face_utils.extract_face(frame, (x, y, w, h))
                        
                        # Recognize face
                        student_id, confidence = face_utils.recognize_face(
                            face_img, 
                            config.TRAINER_PATH,
                            config.FACE_DETECTION_CONFIDENCE
                        )
                        
                        # Draw rectangle and name
                        if student_id != -1:
                            # Get student name
                            student_name = db_utils.get_student_name(student_id)
                            if student_name:
                                # Mark attendance
                                if student_id not in recognized_students:
                                    db_utils.mark_attendance(student_id, subject)
                                    recognized_students.add(student_id)
                                
                                # Draw green rectangle for recognized face
                                cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(
                                    frame_display,
                                    f"{student_name} ({student_id})",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2
                                )
                            else:
                                # Draw yellow rectangle for unknown student ID
                                cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 255), 2)
                                cv2.putText(
                                    frame_display,
                                    f"Unknown ID: {student_id}",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 255),
                                    2
                                )
                        else:
                            # Draw red rectangle for unrecognized face
                            cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(
                                frame_display,
                                "Unknown",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2
                            )
                    
                    # Display the frame
                    cv2.imshow("Attendance", frame_display)
                    
                    # Check for key press
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                
                # Release resources
                cam.release()
                cv2.destroyAllWindows()
                
                # Update message
                if recognized_students:
                    self.root.after(0, lambda: self.attendance_message.config(
                        text=f"Attendance taken for {len(recognized_students)} students."
                    ))
                    text_to_speech(f"Attendance taken for {len(recognized_students)} students.")
                else:
                    self.root.after(0, lambda: self.attendance_message.config(
                        text="No students recognized."
                    ))
                    text_to_speech("No students recognized.")
                
                # Update status
                self.root.after(0, lambda: self.status_bar.config(text="Ready"))
                
            except Exception as e:
                error_msg = f"Error taking attendance: {str(e)}"
                self.root.after(0, lambda: self.attendance_message.config(text=error_msg))
                text_to_speech("Error taking attendance.")
                print(error_msg)
        
        # Run the attendance task in a separate thread
        threading.Thread(target=attendance_task, daemon=True).start()
    
    def view_attendance_button_action(self):
        """Action for View Attendance button"""
        subject = self.subject_entry.get()
        
        if not subject:
            ui_utils.show_message(
                self.root,
                "Error",
                "Please enter a subject name.",
                "error"
            )
            text_to_speech("Please enter a subject name.")
            return
        
        # Get today's date
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Get attendance data
        attendance_df = db_utils.get_attendance(subject, date)
        
        if attendance_df.empty:
            ui_utils.show_message(
                self.root,
                "Info",
                f"No attendance records found for {subject} on {date}.",
                "info"
            )
            text_to_speech(f"No attendance records found for {subject} on {date}.")
            return
        
        # Create a new window to display attendance
        attendance_window = Toplevel(self.root)
        attendance_window.title(f"Attendance for {subject} - {date}")
        attendance_window.geometry("800x600")
        attendance_window.configure(background=config.UI_THEME["bg_color"])
        
        # Create a frame for the attendance table
        table_frame = tk.Frame(attendance_window, bg=config.UI_THEME["bg_color"])
        table_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        # Create a treeview widget
        import tkinter.ttk as ttk
        
        # Configure the treeview style
        style = ttk.Style()
        style.configure(
            "Treeview",
            background=config.UI_THEME["bg_color"],
            foreground=config.UI_THEME["fg_color"],
            rowheight=25,
            fieldbackground=config.UI_THEME["bg_color"]
        )
        style.map(
            "Treeview",
            background=[("selected", config.UI_THEME["highlight_bg"])]
        )
        
        # Create the treeview
        tree = ttk.Treeview(table_frame)
        tree["columns"] = ("Enrollment", "Name", "Time")
        
        # Configure columns
        tree.column("#0", width=0, stretch=NO)
        tree.column("Enrollment", anchor=CENTER, width=150)
        tree.column("Name", anchor=W, width=300)
        tree.column("Time", anchor=CENTER, width=150)
        
        # Configure headings
        tree.heading("#0", text="", anchor=CENTER)
        tree.heading("Enrollment", text="Enrollment", anchor=CENTER)
        tree.heading("Name", text="Name", anchor=CENTER)
        tree.heading("Time", text="Time", anchor=CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Add data to the treeview
        for i, row in attendance_df.iterrows():
            tree.insert(
                "",
                END,
                values=(row["Enrollment"], row["Name"], row["Time"])
            )
        
        # Pack the treeview
        tree.pack(fill=BOTH, expand=True)
        
        # Add a close button
        close_button = ui_utils.create_rounded_button(
            attendance_window,
            "Close",
            attendance_window.destroy,
            width=10,
            height=1
        )
        close_button.pack(pady=10)

# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop() 