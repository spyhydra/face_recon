import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import threading
from tqdm import tqdm
from sklearn.svm import SVC
import joblib

# Import our utility modules
import config
import advanced_face_recognition as afr
import ui_utils

def train_advanced_model(message, text_to_speech, parent_window=None):
    """
    Train the advanced face recognition model
    
    Args:
        message: Label to update with status messages
        text_to_speech: Function to speak messages
        parent_window: Parent window for message boxes
        
    Returns:
        True if training was successful, False otherwise
    """
    try:
        # Check if training directory exists
        if not os.path.exists(config.TRAINING_DIR):
            error_msg = f"Training directory does not exist: {config.TRAINING_DIR}"
            message.config(text=error_msg)
            text_to_speech("Training directory not found.")
            return False
        
        # Check if there are any student directories
        student_dirs = [d for d in os.listdir(config.TRAINING_DIR) 
                       if os.path.isdir(os.path.join(config.TRAINING_DIR, d)) and d != ".git"]
        
        if not student_dirs:
            error_msg = "No student directories found. Please register students first."
            message.config(text=error_msg)
            text_to_speech("No student directories found. Please register students first.")
            return False
        
        # Update message
        info_msg = "Training advanced model. This may take a few minutes..."
        message.config(text=info_msg)
        text_to_speech(info_msg)
        
        # Define a function to run the training in a separate thread
        def train_model_task():
            try:
                # Train the advanced model
                model_path = os.path.join(config.TRAINING_IMAGE_LABEL_DIR, "advanced_model.pkl")
                
                # Check if we only have one person registered
                if len(student_dirs) == 1:
                    # For single-person models, use a special training approach
                    result = train_single_person_model(config.TRAINING_DIR, model_path, student_dirs[0])
                else:
                    # For multiple people, use the standard training
                    result = afr.train_advanced_model(
                        config.TRAINING_DIR,
                        model_path
                    )
                
                if result:
                    success_msg = "Advanced model trained successfully!"
                    message.config(text=success_msg)
                    text_to_speech(success_msg)
                    return True
                else:
                    error_msg = "Failed to train advanced model. Please try again."
                    message.config(text=error_msg)
                    text_to_speech("Failed to train advanced model.")
                    return False
            except Exception as e:
                error_msg = f"Error training advanced model: {str(e)}"
                message.config(text=error_msg)
                text_to_speech("Error training advanced model.")
                print(error_msg)
                return False
        
        # Run the training task in a separate thread
        threading.Thread(target=train_model_task, daemon=True).start()
        return True
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        message.config(text=error_msg)
        text_to_speech("Error occurred.")
        print(error_msg)
        return False

def train_single_person_model(train_dir, model_path, person_dir_name):
    """
    Train a more robust single-person model with enhanced negative samples
    
    Args:
        train_dir: Directory containing training images
        model_path: Path to save the trained model
        person_dir_name: Name of the directory for the single person
        
    Returns:
        True if training was successful, False otherwise
    """
    try:
        print("Using enhanced single-person training mode...")
        
        # Get the person directory
        person_dir = os.path.join(train_dir, person_dir_name)
        
        # Extract enrollment ID from directory name
        if "_" not in person_dir_name:
            print(f"Invalid directory name format: {person_dir_name}")
            return False
        
        try:
            person_id = int(person_dir_name.split("_")[0])
        except ValueError:
            print(f"Could not extract ID from directory name: {person_dir_name}")
            return False
        
        # Get all images for this person
        image_paths = [os.path.join(person_dir, f) for f in os.listdir(person_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_paths:
            print(f"No images found in {person_dir}")
            return False
        
        # Process each image to extract face encodings
        face_encodings = []
        for image_path in image_paths:
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Resize image for faster processing
                height, width = image.shape[:2]
                if width > 640:
                    scale = 640 / width
                    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                
                # Detect face
                face_locations = afr.detect_faces_dlib(image)
                if not face_locations:
                    continue
                
                # Use the largest face if multiple faces detected
                if len(face_locations) > 1:
                    largest_area = 0
                    largest_face_idx = 0
                    for i, (top, right, bottom, left) in enumerate(face_locations):
                        area = (bottom - top) * (right - left)
                        if area > largest_area:
                            largest_area = area
                            largest_face_idx = i
                    face_location = face_locations[largest_face_idx]
                else:
                    face_location = face_locations[0]
                
                # Extract face encoding
                encoding = afr.extract_face_encoding(image, face_location)
                if encoding is not None:
                    face_encodings.append(encoding)
            
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
        
        if len(face_encodings) < 3:
            print("Not enough valid face encodings found (minimum 3 required)")
            return False
        
        # Create training data
        X = np.array(face_encodings)
        y = np.array([person_id] * len(face_encodings))
        
        # Create synthetic negative samples with various transformations
        X_augmented = X.copy()
        y_augmented = y.copy()
        
        # 1. Add random noise samples
        np.random.seed(42)
        for i in range(min(len(X), 10)):
            noise = np.random.normal(0, 0.1, X[i].shape)
            negative_sample = X[i] + noise
            negative_sample = negative_sample / np.linalg.norm(negative_sample)
            X_augmented = np.vstack([X_augmented, negative_sample])
            y_augmented = np.append(y_augmented, -100)
        
        # 2. Add more extreme variations
        for i in range(min(len(X), 5)):
            # More extreme noise
            noise = np.random.normal(0, 0.2, X[i].shape)
            negative_sample = X[i] + noise
            negative_sample = negative_sample / np.linalg.norm(negative_sample)
            X_augmented = np.vstack([X_augmented, negative_sample])
            y_augmented = np.append(y_augmented, -100)
            
            # Reversed features (like a negative image)
            negative_sample = -X[i]
            negative_sample = negative_sample / np.linalg.norm(negative_sample)
            X_augmented = np.vstack([X_augmented, negative_sample])
            y_augmented = np.append(y_augmented, -100)
        
        # 3. Add completely random encodings
        for i in range(15):
            random_encoding = np.random.normal(0, 1, X[0].shape)
            random_encoding = random_encoding / np.linalg.norm(random_encoding)
            X_augmented = np.vstack([X_augmented, random_encoding])
            y_augmented = np.append(y_augmented, -100)
        
        # Train with augmented data
        print(f"Training with {len(X_augmented)} face encodings ({len(X)} real, {len(X_augmented)-len(X)} synthetic)")
        
        # Use a more robust classifier for single-person models
        classifier = SVC(C=1.0, kernel='linear', probability=True, class_weight='balanced')
        classifier.fit(X_augmented, y_augmented)
        
        # Save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(classifier, model_path)
        
        # Also save as face_classifier.pkl for the module to use
        classifier_path = os.path.join(config.TRAINING_IMAGE_LABEL_DIR, "face_classifier.pkl")
        joblib.dump(classifier, classifier_path)
        
        print(f"Enhanced single-person model trained and saved to {model_path}")
        return True
    
    except Exception as e:
        print(f"Error training enhanced single-person model: {e}")
        return False

if __name__ == "__main__":
    # Create a simple UI for testing
    root = tk.Tk()
    root.title("Train Advanced Model")
    root.geometry("400x200")
    
    # Create message label
    message = tk.Label(root, text="Click the button to train the advanced model")
    message.pack(pady=20)
    
    # Simple text-to-speech function for testing
    def simple_tts(text):
        print(f"TTS: {text}")
    
    # Create train button
    train_button = tk.Button(
        root, 
        text="Train Advanced Model", 
        command=lambda: train_advanced_model(message, simple_tts, root)
    )
    train_button.pack(pady=20)
    
    root.mainloop() 