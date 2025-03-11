import os
import cv2
import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import config
from skimage.metrics import structural_similarity as ssim
import face_recognition

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

def check_face_exists(face_image):
    """
    Check if a similar face already exists in the training directory
    
    Args:
        face_image: Face image to check
        
    Returns:
        (exists, student_id, similarity) - Boolean if face exists, student ID if found, and similarity score
    """
    try:
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_image.copy()
            
        # Resize to standard size
        gray_face = cv2.resize(gray_face, (200, 200))
        
        # Apply histogram equalization for better comparison
        gray_face = cv2.equalizeHist(gray_face)
        
        # Get all student directories
        training_dir = config.TRAINING_DIR
        if not os.path.exists(training_dir):
            return False, None, 0
            
        student_dirs = [d for d in os.listdir(training_dir) 
                      if os.path.isdir(os.path.join(training_dir, d))]
        
        if not student_dirs:
            return False, None, 0
            
        # Check similarity with existing faces
        best_match_score = 0
        best_match_id = None
        
        # Dictionary to track average similarity per student
        student_similarities = {}
        
        for student_dir in student_dirs:
            # Extract student ID from directory name
            try:
                student_id = student_dir.split('_')[0]
            except:
                continue
                
            # Get image files in the directory
            student_dir_path = os.path.join(training_dir, student_dir)
            image_files = [f for f in os.listdir(student_dir_path) 
                         if f.endswith('.jpg') or f.endswith('.png')]
            
            # Skip if no images
            if not image_files:
                continue
            
            # Track scores for this student
            scores = []
            
            # Check similarity with each image
            for img_file in image_files:
                img_path = os.path.join(student_dir_path, img_file)
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                        
                    img = cv2.resize(img, (200, 200))
                    img = cv2.equalizeHist(img)
                    
                    # Calculate similarity score using SSIM (Structural Similarity Index)
                    score = ssim(gray_face, img)
                    scores.append(score)
                    
                    # Update best match if better
                    if score > best_match_score:
                        best_match_score = score
                        best_match_id = student_id
                        
                        # If we have a very high similarity with any single image, return immediately
                        if score > 0.8:
                            print(f"Found very similar face with score {score:.4f}")
                            return True, student_id, score
                        
                except Exception as e:
                    print(f"Error comparing with image {img_path}: {e}")
                    continue
            
            # Calculate average similarity for this student (if we have at least 3 scores)
            if len(scores) >= 3:
                # Sort scores and take top 3 (best matches)
                scores.sort(reverse=True)
                top_scores = scores[:3]
                avg_score = sum(top_scores) / len(top_scores)
                student_similarities[student_id] = avg_score
        
        # Find the student with the highest average similarity
        if student_similarities:
            best_student_id = max(student_similarities, key=student_similarities.get)
            best_avg_score = student_similarities[best_student_id]
            
            # Lower the threshold to 0.7 to be more strict
            if best_avg_score > 0.7:
                print(f"Found similar face with average score {best_avg_score:.4f}")
                return True, best_student_id, best_avg_score
        
        # If no good average match, check if we have a strong single match
        if best_match_score > 0.75:  # Lower this threshold too
            print(f"Found similar face with single score {best_match_score:.4f}")
            return True, best_match_id, best_match_score
            
        return False, None, best_match_score if best_match_score > 0 else 0
            
    except Exception as e:
        print(f"Error checking if face exists: {e}")
        return False, None, 0

def check_face_exists_advanced(face_image):
    """
    Check if a similar face already exists using face_recognition library
    
    Args:
        face_image: Face image to check (BGR format)
        
    Returns:
        (exists, student_id, similarity) - Boolean if face exists, student ID if found, and similarity score
    """
    try:
        # Check if face_image is valid
        if face_image is None or face_image.size == 0:
            print("Invalid face image provided to check_face_exists_advanced")
            return False, None, 0
        
        # Convert BGR to RGB for face_recognition library
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Get face encoding
        face_locations = face_recognition.face_locations(rgb_face)
        if not face_locations:
            print("No face detected in the image")
            return False, None, 0
        
        # Use the first face found
        face_encoding = face_recognition.face_encodings(rgb_face, [face_locations[0]])[0]
        
        # Get all student directories
        training_dir = config.TRAINING_DIR
        if not os.path.exists(training_dir):
            return False, None, 0
            
        student_dirs = [d for d in os.listdir(training_dir) 
                      if os.path.isdir(os.path.join(training_dir, d))]
        
        if not student_dirs:
            return False, None, 0
        
        # Check similarity with existing faces
        best_match_score = 0
        best_match_id = None
        
        for student_dir in student_dirs:
            # Extract student ID from directory name
            try:
                student_id = student_dir.split('_')[0]
            except:
                continue
                
            # Get image files in the directory
            student_dir_path = os.path.join(training_dir, student_dir)
            image_files = [f for f in os.listdir(student_dir_path) 
                         if f.endswith('.jpg') or f.endswith('.png')]
            
            # Skip if no images
            if not image_files:
                continue
            
            # Check similarity with each image
            matches_count = 0
            total_similarity = 0
            
            for img_file in image_files[:10]:  # Limit to first 10 images for performance
                img_path = os.path.join(student_dir_path, img_file)
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Convert to RGB
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Get face locations
                    img_face_locations = face_recognition.face_locations(rgb_img)
                    if not img_face_locations:
                        continue
                    
                    # Get face encodings
                    img_face_encoding = face_recognition.face_encodings(rgb_img, [img_face_locations[0]])[0]
                    
                    # Compare faces
                    face_distance = face_recognition.face_distance([img_face_encoding], face_encoding)[0]
                    similarity = 1.0 - min(face_distance, 1.0)
                    
                    # If very similar, count as match
                    if similarity > 0.6:  # Lower threshold for face_recognition
                        matches_count += 1
                        total_similarity += similarity
                    
                    # Update best match
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_id = student_id
                        
                        # If extremely similar, return immediately
                        if similarity > 0.8:
                            print(f"Found very similar face with advanced score {similarity:.4f}")
                            return True, student_id, similarity
                    
                except Exception as e:
                    print(f"Error in advanced face comparison for {img_path}: {e}")
                    continue
            
            # If we have multiple matches for this student, it's likely the same person
            if matches_count >= 3:
                avg_similarity = total_similarity / matches_count
                print(f"Found {matches_count} matches with average similarity {avg_similarity:.4f}")
                return True, student_id, avg_similarity
        
        # If best match is good enough, return it
        if best_match_score > 0.65:
            print(f"Found similar face with advanced score {best_match_score:.4f}")
            return True, best_match_id, best_match_score
        
        return False, None, best_match_score
        
    except Exception as e:
        print(f"Error in check_face_exists_advanced: {e}")
        return False, None, 0 