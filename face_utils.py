import os
import cv2
import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import config

def detect_faces(image):
    """
    Detect faces in an image using Haar Cascade with additional validation
    
    Args:
        image: Input image (BGR format from OpenCV)
        
    Returns:
        List of face bounding boxes in format (x, y, w, h)
    """
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            print("Invalid image provided to detect_faces")
            return []
            
        # Make a copy of the image to avoid modifying the original
        img = image.copy()
            
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            if img.shape[2] == 3:  # BGR image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:  # BGRA image
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                print(f"Unexpected number of channels: {img.shape[2]}")
                return []
        elif len(img.shape) == 2:
            # Already grayscale
            gray = img
        else:
            print(f"Unexpected image shape: {img.shape}")
            return []
        
        # Load cascades with error checking
        face_cascade = cv2.CascadeClassifier(config.HAARCASCADE_FRONTAL_FACE_PATH)
        if face_cascade.empty():
            print("Error: Could not load face cascade classifier")
            # Try alternate cascade file
            face_cascade = cv2.CascadeClassifier(config.HAARCASCADE_ALT_PATH)
            if face_cascade.empty():
                print("Error: Could not load alternate face cascade classifier")
                return []
            
        # Detect faces with parameters from config
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=config.FACE_DETECTION_SCALE_FACTOR,
            minNeighbors=config.FACE_DETECTION_MIN_NEIGHBORS,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return []
            
        validated_faces = []
        height, width = gray.shape[:2]
        
        for (x, y, w, h) in faces:
            # Ensure coordinates are within image bounds
            if x < 0 or y < 0 or x + w > width or y + h > height:
                continue
                
            # Add margin to face box
            margin = int(0.1 * max(w, h))
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(w + 2 * margin, width - x)
            h = min(h + 2 * margin, height - y)
            validated_faces.append((x, y, w, h))
        
        return validated_faces
        
    except Exception as e:
        print(f"Error in detect_faces: {e}")
        return []

def extract_face(image, face_box):
    """
    Extract and preprocess face from image
    
    Args:
        image: Input image
        face_box: Face bounding box (x, y, w, h)
        
    Returns:
        Preprocessed face image
    """
    try:
        x, y, w, h = face_box
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        
        # Extract face region
        face = image[y:y+h, x:x+w]
        
        # Convert to grayscale if needed
        if len(face.shape) == 3:
            if face.shape[2] == 3:  # BGR
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            elif face.shape[2] == 4:  # BGRA
                face = cv2.cvtColor(face, cv2.COLOR_BGRA2GRAY)
        
        # Resize to standard size
        face = cv2.resize(face, (200, 200))
        
        # Enhance contrast
        face = cv2.equalizeHist(face)
        
        return face
    except Exception as e:
        print(f"Error extracting face: {e}")
        return None

def is_face_already_registered(face_img, train_dir, similarity_threshold=0.8):
    """
    Check if a face is already registered in the training directory
    
    Args:
        face_img: Face image to check
        train_dir: Directory containing training images
        similarity_threshold: Threshold for face similarity (0-1)
        
    Returns:
        (is_registered, person_name) tuple
    """
    try:
        # Check if face_img is valid
        if face_img is None or face_img.size == 0:
            print("Invalid face image provided to is_face_already_registered")
            return False, None
            
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            if face_img.shape[2] == 3:  # BGR
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            elif face_img.shape[2] == 4:  # BGRA
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2GRAY)
        
        # Ensure face_img is 200x200
        if face_img.shape[0] != 200 or face_img.shape[1] != 200:
            face_img = cv2.resize(face_img, (200, 200))
        
        # Normalize input face
        face_img = cv2.equalizeHist(face_img)
        
        # Check if train_dir exists
        if not os.path.exists(train_dir):
            print(f"Training directory does not exist: {train_dir}")
            return False, None
            
        # Get all person directories
        person_dirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))]
        
        # Compare with each existing face
        for person_dir in person_dirs:
            try:
                person_name = os.path.basename(person_dir).split("_", 1)[1]  # Get name part after enrollment_
                
                # Get all images for this person
                image_paths = [os.path.join(person_dir, f) for f in os.listdir(person_dir) 
                             if f.endswith('.jpg')]
                
                for image_path in image_paths:
                    try:
                        # Load existing face
                        existing_face = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if existing_face is None:
                            print(f"Could not load image: {image_path}")
                            continue
                            
                        # Resize if needed
                        if existing_face.shape[0] != 200 or existing_face.shape[1] != 200:
                            existing_face = cv2.resize(existing_face, (200, 200))
                            
                        # Enhance contrast
                        existing_face = cv2.equalizeHist(existing_face)
                        
                        # Compare faces using normalized correlation coefficient
                        result = cv2.matchTemplate(
                            face_img, 
                            existing_face, 
                            cv2.TM_CCORR_NORMED
                        )
                        
                        similarity = result[0][0]
                        
                        if similarity > similarity_threshold:
                            return True, person_name
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
                        continue
            except Exception as e:
                print(f"Error processing person directory {person_dir}: {e}")
                continue
        
        return False, None
        
    except Exception as e:
        print(f"Error checking face registration: {e}")
        return False, None

def train_model(train_dir, model_path):
    """
    Train a face recognition model using images in the training directory
    
    Args:
        train_dir: Directory containing training images
        model_path: Path to save the trained model
        
    Returns:
        True if training was successful, False otherwise
    """
    try:
        # Check if training directory exists
        if not os.path.exists(train_dir):
            print(f"Training directory does not exist: {train_dir}")
            return False
            
        # Get all person directories
        person_dirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))]
        
        if not person_dirs:
            print("No person directories found in training directory")
            return False
            
        faces = []  # Will store all face images
        ids = []    # Will store corresponding IDs
        
        for person_dir in tqdm(person_dirs, desc="Processing training data"):
            try:
                # Extract enrollment ID from directory name
                dir_name = os.path.basename(person_dir)
                if "_" not in dir_name:
                    print(f"Invalid directory name format: {dir_name}")
                    continue
                    
                try:
                    person_id = int(dir_name.split("_")[0])
                except ValueError:
                    print(f"Could not extract ID from directory name: {dir_name}")
                    continue
                
                # Get all images for this person
                image_paths = [os.path.join(person_dir, f) for f in os.listdir(person_dir) 
                              if f.endswith('.jpg')]
                
                if not image_paths:
                    print(f"No images found in {person_dir}")
                    continue
                
                person_faces = []
                for image_path in image_paths:
                    try:
                        # Load image as grayscale
                        pil_image = Image.open(image_path).convert('L')
                        # Resize to a standard size
                        pil_image = pil_image.resize((200, 200))
                        image_np = np.array(pil_image, 'uint8')
                        
                        # Enhance contrast
                        image_np = cv2.equalizeHist(image_np)
                        
                        # Add to training data
                        person_faces.append(image_np)
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
                        continue
                
                # Only use this person if we have enough valid images
                if len(person_faces) >= 5:  # Require at least 5 valid images
                    faces.extend(person_faces)
                    ids.extend([person_id] * len(person_faces))
                else:
                    print(f"Not enough valid images for person {dir_name}, skipping")
            except Exception as e:
                print(f"Error processing person directory {person_dir}: {e}")
                continue
        
        if not faces:
            print("No training data found!")
            return False
            
        if len(faces) < 10:  # Require at least 10 total face images
            print(f"Not enough training data: only {len(faces)} images found")
            return False
            
        print(f"Training with {len(faces)} images for {len(set(ids))} people")
            
        # Create LBPH recognizer with optimized parameters
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,           # Default is 1
            neighbors=8,        # Default is 8
            grid_x=8,           # Default is 8
            grid_y=8,           # Default is 8
            threshold=config.FACE_DETECTION_CONFIDENCE  # Use our confidence threshold
        )
        
        # Train the model
        print(f"Training model with {len(faces)} images...")
        recognizer.train(faces, np.array(ids))
        
        # Save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        recognizer.write(model_path)
        
        # Verify the model
        try:
            test_recognizer = cv2.face.LBPHFaceRecognizer_create()
            test_recognizer.read(model_path)
            # Test with a sample face
            if faces:
                test_recognizer.predict(faces[0])
                print("Model verification successful")
            else:
                print("No faces available for model verification")
        except Exception as e:
            print(f"Model verification failed: {e}")
            return False
        
        print(f"Model trained and saved to {model_path}")
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def recognize_face(image, model_path, confidence_threshold=None):
    """
    Recognize a face using the trained model
    
    Args:
        image: Input face image (grayscale)
        model_path: Path to the trained model
        confidence_threshold: Confidence threshold for recognition (uses config value if None)
        
    Returns:
        (id, confidence) if face is recognized, (-1, 0) otherwise
    """
    try:
        # Use config threshold if not specified
        if confidence_threshold is None:
            confidence_threshold = config.FACE_DETECTION_CONFIDENCE
            
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Model file does not exist: {model_path}")
            return -1, 0
            
        # Check if image is valid
        if image is None or image.size == 0:
            print("Invalid image provided to recognize_face")
            return -1, 0
            
        # Make a copy to avoid modifying the original
        img = image.copy()
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            if img.shape[2] == 3:  # BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:  # BGRA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                print(f"Unexpected number of channels in image: {img.shape[2]}")
                return -1, 0
        elif len(img.shape) != 2:
            print(f"Unexpected image shape: {img.shape}")
            return -1, 0
            
        # Resize to match training size
        img = cv2.resize(img, (200, 200))
        
        # Enhance contrast
        img = cv2.equalizeHist(img)
        
        # Load the recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            recognizer.read(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return -1, 0
        
        # Predict
        try:
            id, confidence = recognizer.predict(img)
            print(f"Face recognition result: ID={id}, Confidence={confidence}")
            
            # Lower confidence means better match in LBPH
            if confidence < confidence_threshold:
                return id, confidence
            else:
                return -1, 0
        except Exception as e:
            print(f"Error during face prediction: {e}")
            return -1, 0
    except Exception as e:
        print(f"Error recognizing face: {e}")
        return -1, 0

def fallback_detect_faces(image):
    """
    Fallback face detection using basic Haar Cascade without additional validation
    
    Args:
        image: Input image (BGR format from OpenCV)
        
    Returns:
        List of face bounding boxes in format (x, y, w, h)
    """
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            print("Invalid image provided to fallback_detect_faces")
            return []
            
        # Make a copy to avoid modifying the original
        img = image.copy()
            
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            if img.shape[2] == 3:  # BGR image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:  # BGRA image
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                print(f"Unexpected number of channels: {img.shape[2]}")
                return []
        elif len(img.shape) == 2:
            # Already grayscale
            gray = img
        else:
            print(f"Unexpected image shape: {img.shape}")
            return []
        
        # Load cascade with error checking
        face_cascade = cv2.CascadeClassifier(config.HAARCASCADE_FRONTAL_FACE_PATH)
        if face_cascade.empty():
            print("Error: Could not load face cascade classifier")
            return []
            
        # Detect faces with more lenient parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return []
            
        # Add margin to face boxes
        face_boxes = []
        height, width = gray.shape[:2]
        
        for (x, y, w, h) in faces:
            # Add margin to face box
            margin = int(0.1 * max(w, h))
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(w + 2 * margin, width - x)
            h = min(h + 2 * margin, height - y)
            face_boxes.append((x, y, w, h))
        
        return face_boxes
        
    except Exception as e:
        print(f"Error in fallback_detect_faces: {e}")
        return [] 