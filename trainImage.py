import os
import cv2
import numpy as np
import threading
from tqdm import tqdm

# Import our utility modules
import config
import face_utils
import ui_utils

def TrainImage(message, text_to_speech, parent_window=None):
    """
    Train the face recognition model
    
    Args:
        message: Label widget to display messages
        text_to_speech: Function to convert text to speech
        parent_window: Parent window for progress bar
    
    Returns:
        True if training was successful, False otherwise
    """
    # Check if training directory exists and has data
    if not os.path.exists(config.TRAINING_DIR):
        error_msg = "Training directory does not exist."
        message.configure(text=error_msg)
        text_to_speech(error_msg)
        return False
    
    # Check if there are any student directories
    student_dirs = [d for d in os.listdir(config.TRAINING_DIR) 
                   if os.path.isdir(os.path.join(config.TRAINING_DIR, d))]
    
    if not student_dirs:
        error_msg = "No student data found. Please register students first."
        message.configure(text=error_msg)
        text_to_speech(error_msg)
        return False
    
    # Update message
    info_msg = "Training model. This may take a few minutes..."
    message.configure(text=info_msg)
    text_to_speech(info_msg)
    
    # Define the training function
    def train_model_task():
        try:
            # Train the model
            result = face_utils.train_model(
                config.TRAINING_DIR,
                config.TRAINER_PATH
            )
            
            # Update message based on result
            if result:
                success_msg = "Model trained successfully!"
                message.configure(text=success_msg)
                text_to_speech(success_msg)
                return True
            else:
                error_msg = "Failed to train model. Please try again."
                message.configure(text=error_msg)
                text_to_speech(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Error training model: {str(e)}"
            message.configure(text=error_msg)
            text_to_speech(error_msg)
            print(error_msg)
            return False
    
    # Run the training task with a progress bar if parent window is provided
    if parent_window:
        return ui_utils.run_with_progress(
            parent_window,
            train_model_task,
            "Training face recognition model..."
        )
    else:
        return train_model_task()
