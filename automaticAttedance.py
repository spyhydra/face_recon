import tkinter as tk
from tkinter import *
import os, cv2
import shutil
import csv
import numpy as np
from PIL import ImageTk, Image
import pandas as pd
import datetime
import time
import tkinter.ttk as tkk
import tkinter.font as font
import threading

# Import our utility modules
import config
import face_utils
import db_utils
import ui_utils

haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = (
    "TrainingImageLabel\\Trainner.yml"
)
trainimage_path = "TrainingImage"
studentdetail_path = (
    "StudentDetails\\studentdetails.csv"
)
attendance_path = "Attendance"
# for choose subject and fill attendance
def subjectChoose(text_to_speech):
    def FillAttendance():
        sub = tx.get()
        now = time.time()
        future = now + 20
        print(now)
        print(future)
        if sub == "":
            t = "Please enter the subject name!!!"
            text_to_speech(t)
        else:
            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                try:
                    recognizer.read(trainimagelabel_path)
                except:
                    e = "Model not found,please train model"
                    Notifica.configure(
                        text=e,
                        bg="black",
                        fg="yellow",
                        width=33,
                        font=("times", 15, "bold"),
                    )
                    Notifica.place(x=20, y=250)
                    text_to_speech(e)
                facecasCade = cv2.CascadeClassifier(haarcasecade_path)
                df = pd.read_csv(studentdetail_path)
                cam = cv2.VideoCapture(0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                col_names = ["Enrollment", "Name"]
                attendance = pd.DataFrame(columns=col_names)
                while True:
                    ___, im = cam.read()
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces = facecasCade.detectMultiScale(gray, 1.2, 5)
                    for (x, y, w, h) in faces:
                        global Id

                        Id, conf = recognizer.predict(gray[y : y + h, x : x + w])
                        if conf < 70:
                            print(conf)
                            global Subject
                            global aa
                            global date
                            global timeStamp
                            Subject = tx.get()
                            ts = time.time()
                            date = datetime.datetime.fromtimestamp(ts).strftime(
                                "%Y-%m-%d"
                            )
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime(
                                "%H:%M:%S"
                            )
                            aa = df.loc[df["Enrollment"] == Id]["Name"].values
                            global tt
                            tt = str(Id) + "-" + aa
                            # En='1604501160'+str(Id)
                            attendance.loc[len(attendance)] = [
                                Id,
                                aa,
                            ]
                            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 4)
                            cv2.putText(
                                im, str(tt), (x + h, y), font, 1, (255, 255, 0,), 4
                            )
                        else:
                            Id = "Unknown"
                            tt = str(Id)
                            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 7)
                            cv2.putText(
                                im, str(tt), (x + h, y), font, 1, (0, 25, 255), 4
                            )
                    if time.time() > future:
                        break

                    attendance = attendance.drop_duplicates(
                        ["Enrollment"], keep="first"
                    )
                    cv2.imshow("Filling Attendance...", im)
                    key = cv2.waitKey(30) & 0xFF
                    if key == 27:
                        break

                ts = time.time()
                print(aa)
                # attendance["date"] = date
                # attendance["Attendance"] = "P"
                attendance[date] = 1
                date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                Hour, Minute, Second = timeStamp.split(":")
                # fileName = "Attendance/" + Subject + ".csv"
                path = os.path.join(attendance_path, Subject)
                if not os.path.exists(path):
                    os.makedirs(path)
                fileName = (
                    f"{path}/"
                    + Subject
                    + "_"
                    + date
                    + "_"
                    + Hour
                    + "-"
                    + Minute
                    + "-"
                    + Second
                    + ".csv"
                )
                attendance = attendance.drop_duplicates(["Enrollment"], keep="first")
                print(attendance)
                attendance.to_csv(fileName, index=False)

                m = "Attendance Filled Successfully of " + Subject
                Notifica.configure(
                    text=m,
                    bg="black",
                    fg="yellow",
                    width=33,
                    relief=RIDGE,
                    bd=5,
                    font=("times", 15, "bold"),
                )
                text_to_speech(m)

                Notifica.place(x=20, y=250)

                cam.release()
                cv2.destroyAllWindows()

                import csv
                import tkinter

                root = tkinter.Tk()
                root.title("Attendance of " + Subject)
                root.configure(background="black")
                cs = os.path.join(path, fileName)
                print(cs)
                with open(cs, newline="") as file:
                    reader = csv.reader(file)
                    r = 0

                    for col in reader:
                        c = 0
                        for row in col:

                            label = tkinter.Label(
                                root,
                                width=10,
                                height=1,
                                fg="yellow",
                                font=("times", 15, " bold "),
                                bg="black",
                                text=row,
                                relief=tkinter.RIDGE,
                            )
                            label.grid(row=r, column=c)
                            c += 1
                        r += 1
                root.mainloop()
                print(attendance)
            except:
                f = "No Face found for attendance"
                text_to_speech(f)
                cv2.destroyAllWindows()

    ###windo is frame for subject chooser
    subject = Tk()
    # windo.iconbitmap("AMS.ico")
    subject.title("Subject...")
    subject.geometry("580x320")
    subject.resizable(0, 0)
    subject.configure(background="black")
    # subject_logo = Image.open("UI_Image/0004.png")
    # subject_logo = subject_logo.resize((50, 47), Image.ANTIALIAS)
    # subject_logo1 = ImageTk.PhotoImage(subject_logo)
    titl = tk.Label(subject, bg="black", relief=RIDGE, bd=10, font=("arial", 30))
    titl.pack(fill=X)
    # l1 = tk.Label(subject, image=subject_logo1, bg="black",)
    # l1.place(x=100, y=10)
    titl = tk.Label(
        subject,
        text="Enter the Subject Name",
        bg="black",
        fg="green",
        font=("arial", 25),
    )
    titl.place(x=160, y=12)
    Notifica = tk.Label(
        subject,
        text="Attendance filled Successfully",
        bg="yellow",
        fg="black",
        width=33,
        height=2,
        font=("times", 15, "bold"),
    )

    def Attf():
        sub = tx.get()
        if sub == "":
            t = "Please enter the subject name!!!"
            text_to_speech(t)
        else:
            os.startfile(
                f"Attendance\\{sub}"
            )

    attf = tk.Button(
        subject,
        text="Check Sheets",
        command=Attf,
        bd=7,
        font=("times new roman", 15),
        bg="black",
        fg="yellow",
        height=2,
        width=10,
        relief=RIDGE,
    )
    attf.place(x=360, y=170)

    sub = tk.Label(
        subject,
        text="Enter Subject",
        width=10,
        height=2,
        bg="black",
        fg="yellow",
        bd=5,
        relief=RIDGE,
        font=("times new roman", 15),
    )
    sub.place(x=50, y=100)

    tx = tk.Entry(
        subject,
        width=15,
        bd=5,
        bg="black",
        fg="yellow",
        relief=RIDGE,
        font=("times", 30, "bold"),
    )
    tx.place(x=190, y=100)

    fill_a = tk.Button(
        subject,
        text="Fill Attendance",
        command=FillAttendance,
        bd=7,
        font=("times new roman", 15),
        bg="black",
        fg="yellow",
        height=2,
        width=12,
        relief=RIDGE,
    )
    fill_a.place(x=195, y=170)
    subject.mainloop()

def take_attendance(subject, message_label, text_to_speech, parent_window=None):
    """
    Take attendance using face recognition
    
    Args:
        subject: Subject name
        message_label: Label widget to display messages
        text_to_speech: Function to convert text to speech
        parent_window: Parent window for progress bar
    
    Returns:
        True if attendance was taken successfully, False otherwise
    """
    # Check if subject is provided
    if not subject:
        error_msg = "Please enter a subject name."
        message_label.configure(text=error_msg)
        text_to_speech(error_msg)
        return False
    
    # Check if model exists
    if not os.path.exists(config.TRAINER_PATH):
        error_msg = "Model not found. Please train the model first."
        message_label.configure(text=error_msg)
        text_to_speech(error_msg)
        return False
    
    # Update message
    message_label.configure(text="Starting camera...")
    
    # Define the attendance task
    def attendance_task():
        try:
            # Initialize camera
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                error_msg = "Could not open camera. Please check your camera connection."
                message_label.configure(text=error_msg)
                text_to_speech(error_msg)
                return False
            
            # Create a window for displaying the camera feed
            cv2.namedWindow("Attendance", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Attendance", 640, 480)
            
            # Load student data
            students_df = db_utils.get_all_students()
            if students_df.empty:
                error_msg = "No students found. Please register students first."
                message_label.configure(text=error_msg)
                text_to_speech(error_msg)
                cam.release()
                cv2.destroyAllWindows()
                return False
            
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
                
                # Update message
                message_label.configure(text=f"Taking attendance... {time_remaining}s remaining")
                
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
                success_msg = f"Attendance taken for {len(recognized_students)} students."
                message_label.configure(text=success_msg)
                text_to_speech(success_msg)
                return True
            else:
                info_msg = "No students recognized."
                message_label.configure(text=info_msg)
                text_to_speech(info_msg)
                return False
            
        except Exception as e:
            error_msg = f"Error taking attendance: {str(e)}"
            message_label.configure(text=error_msg)
            text_to_speech("Error taking attendance.")
            print(error_msg)
            return False
    
    # Run the attendance task in a separate thread if parent window is not provided
    if parent_window is None:
        threading.Thread(target=attendance_task, daemon=True).start()
        return True
    else:
        # Run with progress bar if parent window is provided
        return ui_utils.run_with_progress(
            parent_window,
            attendance_task,
            "Taking attendance..."
        )
