import os
import cv2
import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import config

def detect_faces(image):
    """
    Detect faces in an image using Haar Cascade
    
    Args:
        image: Input image (BGR format from OpenCV)
        
    Returns:
        List of face bounding boxes in format (x, y, w, h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(config.HAARCASCADE_FRONTAL_FACE_PATH)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=config.FACE_DETECTION_SCALE_FACTOR,
        minNeighbors=config.FACE_DETECTION_MIN_NEIGHBORS
    )
    
    # Add margin to face boxes
    face_boxes = []
    for (x, y, w, h) in faces:
        # Add a margin to the face box
        margin = int(0.1 * max(w, h))
        x = max(0, x - margin)
        y = max(0, y - margin)
        w += 2 * margin
        h += 2 * margin
        face_boxes.append((x, y, w, h))
    
    return face_boxes

def extract_face(image, face_box):
    """
    Extract a face from an image
    
    Args:
        image: Input image
        face_box: Face bounding box (x, y, w, h)
        
    Returns:
        Extracted face image
    """
    x, y, w, h = face_box
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = gray[y:y+h, x:x+w]
    # Resize to a standard size
    face = cv2.resize(face, (200, 200))
    return face

def is_face_already_registered(face_image, train_dir, similarity_threshold=0.8):
    """
    Check if a face is already registered by comparing with existing faces
    
    Args:
        face_image: Input face image (grayscale, 200x200)
        train_dir: Directory containing training images
        similarity_threshold: Threshold for face similarity (0-1)
        
    Returns:
        (bool, str): (is_registered, matching_person_name)
    """
    try:
        # Get all person directories
        person_dirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))]
        
        # Normalize input face
        face_image = cv2.equalizeHist(face_image)
        
        # Compare with each existing face
        for person_dir in person_dirs:
            person_name = os.path.basename(person_dir).split("_", 1)[1]  # Get name part after enrollment_
            
            # Get all images for this person
            image_paths = [os.path.join(person_dir, f) for f in os.listdir(person_dir) 
                         if f.endswith('.jpg')]
            
            for image_path in image_paths:
                # Load and preprocess existing face
                existing_face = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                existing_face = cv2.resize(existing_face, (200, 200))
                existing_face = cv2.equalizeHist(existing_face)
                
                # Compare faces using normalized correlation coefficient
                similarity = cv2.matchTemplate(
                    face_image, 
                    existing_face, 
                    cv2.TM_CCORR_NORMED
                )[0][0]
                
                if similarity > similarity_threshold:
                    return True, person_name
        
        return False, None
        
    except Exception as e:
        print(f"Error checking face registration: {e}")
        return False, None

def train_model(train_dir, model_path):
    """
    Train face recognition model using LBPH
    
    Args:
        train_dir: Directory containing training images
        model_path: Path to save the trained model
        
    Returns:
        True if training was successful, False otherwise
    """
    try:
        # Create LBPH recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Get all person directories
        person_dirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))]
        
        faces = []
        ids = []
        
        # Process each person's images
        for person_dir in tqdm(person_dirs, desc="Processing training data"):
            person_id = int(os.path.basename(person_dir).split("_")[0])
            
            # Get all images for this person
            image_paths = [os.path.join(person_dir, f) for f in os.listdir(person_dir) 
                         if f.endswith('.jpg')]
            
            for image_path in image_paths:
                # Load image and convert to grayscale
                pil_image = Image.open(image_path).convert('L')
                # Resize to a standard size
                pil_image = pil_image.resize((200, 200))
                image_np = np.array(pil_image, 'uint8')
                
                # Add to training data
                faces.append(image_np)
                ids.append(person_id)
        
        if not faces:
            print("No training data found!")
            return False
            
        # Train the model
        print(f"Training model with {len(faces)} images...")
        recognizer.train(faces, np.array(ids))
        
        # Save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        recognizer.write(model_path)
        print(f"Model saved to {model_path}")
        
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def recognize_face(image, model_path, confidence_threshold=70):
    """
    Recognize a face using the trained model
    
    Args:
        image: Input face image (grayscale)
        model_path: Path to the trained model
        confidence_threshold: Confidence threshold for recognition
        
    Returns:
        (id, confidence) if face is recognized, (-1, 0) otherwise
    """
    try:
        # Resize to match training size
        image = cv2.resize(image, (200, 200))
        
        # Load the recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)
        
        # Predict
        id, confidence = recognizer.predict(image)
        
        # Lower confidence means better match in LBPH
        if confidence < confidence_threshold:
            return id, confidence
        else:
            return -1, 0
    except Exception as e:
        print(f"Error recognizing face: {e}")
        return -1, 0 