import os
import cv2
import time
import config
import face_utils
import numpy as np
from PIL import Image

def TakeImage(enrollment, name, message_label, text_to_speech):
    """
    Capture images for a student
    
    Args:
        enrollment: Student enrollment number
        name: Student name
        message_label: Label widget to display messages
        text_to_speech: Function to convert text to speech
        
    Returns:
        True if images captured successfully, False otherwise
    """
    try:
        # Check if student already exists
        import db_utils
        if db_utils.student_exists(enrollment):
            message = f"Worker with ID {enrollment} already exists."
            message_label.config(text=message)
            text_to_speech(message)
            return False
        
        # Create directory for student images
        student_dir = os.path.join(config.TRAINING_DIR, f"{enrollment}_{name}")
        os.makedirs(student_dir, exist_ok=True)
        
        # Initialize camera
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            message = "Could not open camera. Please check your camera connection."
            message_label.config(text=message)
            text_to_speech(message)
            return False
        
        # Set up face detector
        message = "Camera opened successfully. Looking for face..."
        message_label.config(text=message)
        text_to_speech(message)
        
        # Counter for images
        img_counter = 0
        sample_count = config.SAMPLE_COUNT
        
        # Start time for timeout
        start_time = time.time()
        timeout = 120  # 120 seconds timeout
        
        # Time of last capture
        last_capture_time = 0
        capture_delay = 0.5  # 0.5 seconds between captures
        
        # Track if we've found a duplicate
        duplicate_found = False
        
        # Instructions
        cv2.namedWindow("Take Images", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Take Images", 640, 480)
        message_label.config(text="Press SPACE to capture manually or wait for automatic capture. ESC to cancel.")
        text_to_speech("Press SPACE to capture manually or wait for automatic capture. ESC to cancel.")
        
        while img_counter < sample_count:
            # Check for timeout
            if time.time() - start_time > timeout:
                message = "Timeout reached. Please try again."
                message_label.config(text=message)
                text_to_speech(message)
                cam.release()
                cv2.destroyAllWindows()
                # Clean up the created directory
                try:
                    import shutil
                    shutil.rmtree(student_dir)
                except:
                    pass
                return False
            
            # Read frame
            ret, frame = cam.read()
            if not ret:
                message = "Failed to grab frame. Please try again."
                message_label.config(text=message)
                text_to_speech(message)
                cam.release()
                cv2.destroyAllWindows()
                return False
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Detect faces
            faces = face_utils.detect_faces(frame)
            if len(faces) == 0:
                # If no faces detected, try fallback method
                faces = face_utils.fallback_detect_faces(frame)
            
            # Draw rectangle around face and add text
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add text with instructions and progress
            cv2.putText(
                display_frame,
                f"Images: {img_counter}/{sample_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                display_frame,
                "SPACE: Manual capture, ESC: Cancel",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display frame
            cv2.imshow("Take Images", display_frame)
            
            # Wait for key press (1ms delay)
            key = cv2.waitKey(1)
            
            # ESC key to exit
            if key == 27:  # ESC key
                message = "Image capture cancelled."
                message_label.config(text=message)
                text_to_speech(message)
                cam.release()
                cv2.destroyAllWindows()
                # Clean up the created directory
                try:
                    import shutil
                    shutil.rmtree(student_dir)
                except:
                    pass
                return False
            
            # Determine if we should capture now
            capture_now = False
            
            # SPACE key to capture image
            if key == 32 and len(faces) > 0:
                capture_now = True
            # Automatic capture if face detected and enough time has passed
            elif len(faces) > 0 and (time.time() - last_capture_time) > capture_delay:
                capture_now = True
            
            # Capture image if needed
            if capture_now:
                for (x, y, w, h) in faces:
                    # Add margin to face
                    margin = int(0.2 * max(w, h))  # Increased margin for better face capture
                    x_with_margin = max(0, x - margin)
                    y_with_margin = max(0, y - margin)
                    w_with_margin = min(w + 2 * margin, frame.shape[1] - x_with_margin)
                    h_with_margin = min(h + 2 * margin, frame.shape[0] - y_with_margin)
                    
                    # Extract face
                    face = frame[y_with_margin:y_with_margin+h_with_margin, 
                                x_with_margin:x_with_margin+w_with_margin]
                    
                    # Ensure the face image is not empty
                    if face.size == 0:
                        continue
                    
                    # Only check for duplicates for the first few images
                    if img_counter < 3:
                        # Check for duplicates with both methods
                        face_exists = False
                        existing_id = None
                        similarity = 0
                        
                        try:
                            # Try SSIM method first
                            face_exists, existing_id, similarity = face_utils.check_face_exists(face)
                            
                            # If not found with SSIM, try advanced method
                            if not face_exists:
                                face_exists, existing_id, similarity = face_utils.check_face_exists_advanced(face)
                        except Exception as e:
                            print(f"Error in duplicate check: {e}")
                            continue
                        
                        if face_exists and existing_id != enrollment:  # Only consider it duplicate if it's a different ID
                            # Get student name
                            existing_name = db_utils.get_student_name(existing_id)
                            
                            # Show warning message with red text
                            warning_message = f"WARNING: This face is already registered as {existing_name} (ID: {existing_id}) with {similarity*100:.1f}% similarity."
                            message_label.config(text=warning_message, fg="red")
                            text_to_speech("Warning! This worker is already registered in the system.")
                            
                            # Display warning on the image
                            cv2.putText(
                                display_frame,
                                "WORKER ALREADY EXISTS!",
                                (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 0, 255),
                                2
                            )
                            cv2.putText(
                                display_frame,
                                f"Registered as {existing_name} (ID: {existing_id})",
                                (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2
                            )
                            cv2.imshow("Take Images", display_frame)
                            cv2.waitKey(3000)
                            
                            duplicate_found = True
                            break
                    
                    if duplicate_found:
                        break
                    
                    # Save the face image
                    img_name = os.path.join(student_dir, f"{img_counter}.jpg")
                    cv2.imwrite(img_name, face)
                    
                    img_counter += 1
                    last_capture_time = time.time()
                    
                    # Update message
                    message = f"Image {img_counter}/{sample_count} captured."
                    message_label.config(text=message, fg="black")  # Reset text color to black
                    
                    # Only speak every 10 images to avoid too much speech
                    if img_counter % 10 == 0 or img_counter == sample_count:
                        text_to_speech(message)
                    
                    break  # Break after capturing one face
            
            if duplicate_found:
                # Clean up
                cam.release()
                cv2.destroyAllWindows()
                # Remove the directory since we found a duplicate
                try:
                    import shutil
                    shutil.rmtree(student_dir)
                except:
                    pass
                return False
        
        # Release camera and close windows
        cam.release()
        cv2.destroyAllWindows()
        
        # Check if we have enough images
        if img_counter < 5:  # Minimum 5 images required
            message = f"Not enough images captured ({img_counter}). Please try again."
            message_label.config(text=message)
            text_to_speech(message)
            # Clean up the directory
            try:
                import shutil
                shutil.rmtree(student_dir)
            except:
                pass
            return False
        
        # Add student to database only if we have enough images
        if img_counter >= 5:
            db_utils.add_student(enrollment, name)
            
            # Final message
            message = f"Images captured successfully for {name} ({enrollment})."
            message_label.config(text=message)
            text_to_speech(message)
            return True
        
        return False
    
    except Exception as e:
        # Handle exceptions
        error_message = f"Error capturing images: {str(e)}"
        message_label.config(text=error_message)
        text_to_speech("Error capturing images.")
        print(error_message)
        
        # Clean up
        try:
            cam.release()
            cv2.destroyAllWindows()
            # Clean up the directory in case of error
            import shutil
            shutil.rmtree(student_dir)
        except:
            pass
        
        return False

if __name__ == "__main__":
    # For testing
    import tkinter as tk
    
    def dummy_tts(text):
        print(f"TTS: {text}")
    
    root = tk.Tk()
    label = tk.Label(root, text="Testing...")
    label.pack()
    
    TakeImage("101", "Test Student", label, dummy_tts)
    
    root.mainloop() 