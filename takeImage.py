import os
import cv2
import numpy as np
import datetime
import time
import threading
from PIL import Image

# Import our utility modules
import config
import face_utils
import db_utils
import ui_utils

def TakeImage(enrollment, name, message, text_to_speech):
    """
    Take images of a student for training
    
    Args:
        enrollment: Student enrollment number
        name: Student name
        message: Label widget to display messages
        text_to_speech: Function to convert text to speech
    """
    if not enrollment or not name:
        error_msg = "Please enter both Enrollment Number and Name."
        message.configure(text=error_msg)
        text_to_speech(error_msg)
        return False
    
    try:
        # Check if student already exists
        if db_utils.student_exists(enrollment):
            error_msg = f"Student with Enrollment {enrollment} already exists."
            message.configure(text=error_msg)
            text_to_speech(error_msg)
            return False
        
        # Create directory for student images
        directory = f"{enrollment}_{name}"
        path = os.path.join(config.TRAINING_DIR, directory)
        os.makedirs(path, exist_ok=True)
        
        # Initialize camera
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            error_msg = "Could not open camera. Please check your camera connection."
            message.configure(text=error_msg)
            text_to_speech(error_msg)
            return False
        
        # Initialize face detector
        sample_count = 0
        max_samples = config.SAMPLE_COUNT
        
        # Create a window for displaying the camera feed
        cv2.namedWindow("Face Capture", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Capture", 640, 480)
        
        # Display instructions
        info_msg = f"Capturing {max_samples} images. Look at the camera and move your face slightly."
        message.configure(text=info_msg)
        text_to_speech(info_msg)
        
        # Flag to track if face similarity check has been done
        face_checked = False
        
        # Start capturing images
        while True:
            ret, img = cam.read()
            if not ret:
                break
                
            # Display sample count on image
            img_display = img.copy()
            cv2.putText(
                img_display,
                f"Samples: {sample_count}/{max_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Detect faces
            faces = face_utils.detect_faces(img)
            
            # Process detected faces
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(img_display, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Extract face
                face_img = face_utils.extract_face(img, (x, y, w, h))
                
                # Check if this face is already registered (only once)
                if not face_checked:
                    is_registered, existing_name = face_utils.is_face_already_registered(
                        face_img,
                        config.TRAINING_DIR
                    )
                    if is_registered:
                        error_msg = f"This face is already registered under the name: {existing_name}"
                        message.configure(text=error_msg)
                        text_to_speech(error_msg)
                        cam.release()
                        cv2.destroyAllWindows()
                        # Remove the created directory
                        if os.path.exists(path):
                            os.rmdir(path)
                        return False
                    face_checked = True
                
                # Save the face image
                if sample_count < max_samples:
                    sample_count += 1
                    
                    # Save face image
                    filename = os.path.join(
                        path, 
                        f"{name}_{enrollment}_{sample_count}.jpg"
                    )
                    cv2.imwrite(filename, face_img)
                    
                    # Update message
                    message.configure(text=f"Capturing image {sample_count}/{max_samples}")
                    
                    # Add a small delay to avoid duplicate images
                    time.sleep(0.1)
            
            # Display the image
            cv2.imshow("Face Capture", img_display)
            
            # Check for key press or completion
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or sample_count >= max_samples:
                break
        
        # Release resources
        cam.release()
        cv2.destroyAllWindows()
        
        # Check if we captured enough samples
        if sample_count < 10:
            error_msg = f"Only captured {sample_count} images. At least 10 are required."
            message.configure(text=error_msg)
            text_to_speech(error_msg)
            # Remove the created directory
            if os.path.exists(path):
                import shutil
                shutil.rmtree(path)
            return False
        
        # Add student to database
        db_utils.add_student(enrollment, name)
        
        # Display success message
        success_msg = f"Successfully captured {sample_count} images for {name} ({enrollment})."
        message.configure(text=success_msg)
        text_to_speech(success_msg)
        
        return True
        
    except Exception as e:
        error_msg = f"Error capturing images: {str(e)}"
        message.configure(text=error_msg)
        text_to_speech(error_msg)
        print(error_msg)
        # Clean up on error
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        return False
