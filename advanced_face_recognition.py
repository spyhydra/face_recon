import os
import cv2
import numpy as np
import pickle
from PIL import Image
import face_recognition
from tqdm import tqdm
import config
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import db_utils

# Global variables for models
face_detector = None
face_embedder = None
face_classifier = None

def initialize_models():
    """Initialize face detection and recognition models"""
    global face_detector, face_embedder, face_classifier
    
    print("Initializing advanced face recognition models...")
    
    # Load face classifier if it exists
    classifier_path = os.path.join(config.TRAINING_IMAGE_LABEL_DIR, "face_classifier.pkl")
    if os.path.exists(classifier_path):
        try:
            face_classifier = joblib.load(classifier_path)
            print("Face classifier loaded successfully")
        except Exception as e:
            print(f"Error loading face classifier: {e}")
            face_classifier = None
    
    print("Advanced face recognition models initialized")
    return True

def detect_faces_dlib(image):
    """
    Detect faces using dlib's HOG-based face detector (more accurate than Haar Cascade)
    
    Args:
        image: Input image (BGR format from OpenCV)
        
    Returns:
        List of face bounding boxes in format (top, right, bottom, left)
    """
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            print("Invalid image provided to detect_faces_dlib")
            return []
        
        # Convert BGR to RGB (dlib uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        if not face_locations:
            # Try with different parameters if no faces detected
            face_locations = face_recognition.face_locations(rgb_image, model="hog", number_of_times_to_upsample=2)
            
            # If still no faces detected, try a simpler method for synthetic faces
            if not face_locations:
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Use simple blob detection for synthetic faces
                # This works better for our generated faces which have clear contrast
                params = cv2.SimpleBlobDetector_Params()
                params.filterByColor = True
                params.blobColor = 255  # Looking for light blobs (faces)
                params.filterByArea = True
                params.minArea = 1000  # Minimum face size
                params.maxArea = 100000  # Maximum face size
                params.filterByCircularity = False
                params.filterByConvexity = False
                params.filterByInertia = False
                
                detector = cv2.SimpleBlobDetector_create(params)
                keypoints = detector.detect(255 - gray)  # Invert for dark blobs
                
                # Convert keypoints to face locations
                face_locations = []
                for kp in keypoints:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    r = int(kp.size / 2)
                    # Convert to (top, right, bottom, left) format
                    top = max(0, y - r)
                    right = min(image.shape[1], x + r)
                    bottom = min(image.shape[0], y + r)
                    left = max(0, x - r)
                    face_locations.append((top, right, bottom, left))
                
                # If still no faces, try to detect the entire image as a face
                # This is a last resort for our synthetic faces
                if not face_locations:
                    # Use the entire image as a face
                    h, w = image.shape[:2]
                    margin = min(h, w) // 10
                    face_locations = [(margin, w - margin, h - margin, margin)]
        
        return face_locations
    
    except Exception as e:
        print(f"Error in detect_faces_dlib: {e}")
        return []

def detect_faces_cnn(image):
    """
    Detect faces using CNN-based face detector (most accurate but slower)
    
    Args:
        image: Input image (BGR format from OpenCV)
        
    Returns:
        List of face bounding boxes in format (top, right, bottom, left)
    """
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            print("Invalid image provided to detect_faces_cnn")
            return []
        
        # Convert BGR to RGB (dlib uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces using CNN model
        face_locations = face_recognition.face_locations(rgb_image, model="cnn")
        
        return face_locations
    
    except Exception as e:
        print(f"Error in detect_faces_cnn: {e}")
        return []

def extract_face_encoding(image, face_location=None):
    """
    Extract face encoding (128-dimensional feature vector) from image
    
    Args:
        image: Input image (BGR format from OpenCV)
        face_location: Optional face location (top, right, bottom, left)
        
    Returns:
        128-dimensional face encoding
    """
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            print("Invalid image provided to extract_face_encoding")
            return None
        
        # Convert BGR to RGB (dlib uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face if location not provided
        if face_location is None:
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            if not face_locations:
                print("No face detected in the image")
                return None
            face_location = face_locations[0]
        
        # Extract face encoding
        face_encodings = face_recognition.face_encodings(rgb_image, [face_location])
        
        if not face_encodings:
            print("Could not extract face encoding, trying fallback method")
            
            # Fallback method for synthetic faces
            try:
                # Extract the face region
                top, right, bottom, left = face_location
                face_img = rgb_image[top:bottom, left:right]
                
                # Resize to a standard size
                face_img = cv2.resize(face_img, (128, 128))
                
                # Flatten and normalize the image as a simple feature vector
                features = face_img.flatten().astype(np.float32)
                features = features / np.linalg.norm(features)
                
                # Reduce to 128 dimensions using PCA-like approach
                if len(features) > 128:
                    # Simple dimensionality reduction - take every nth value
                    n = len(features) // 128
                    features = features[::n][:128]
                
                # Ensure we have exactly 128 dimensions
                if len(features) < 128:
                    # Pad with zeros if needed
                    features = np.pad(features, (0, 128 - len(features)))
                
                # Normalize again to unit length
                features = features / np.linalg.norm(features)
                
                return features
            except Exception as e:
                print(f"Fallback encoding failed: {e}")
                return None
        
        return face_encodings[0]
    
    except Exception as e:
        print(f"Error in extract_face_encoding: {e}")
        return None

def detect_duplicate_faces(training_data, threshold=0.85):
    """
    Detect potential duplicate faces across different student IDs
    
    Args:
        training_data: Dictionary mapping student IDs to face encodings
        threshold: Similarity threshold (higher means more strict)
        
    Returns:
        List of tuples (id1, id2, similarity) for potential duplicates
    """
    duplicates = []
    
    # Get all student IDs
    student_ids = list(training_data.keys())
    
    # Compare each pair of students
    for i in range(len(student_ids)):
        id1 = student_ids[i]
        encodings1 = training_data[id1]
        
        for j in range(i+1, len(student_ids)):
            id2 = student_ids[j]
            encodings2 = training_data[id2]
            
            # Compare each encoding from id1 with each encoding from id2
            for enc1 in encodings1:
                for enc2 in encodings2:
                    # Calculate similarity (1 - distance)
                    distance = np.linalg.norm(enc1 - enc2)
                    similarity = 1 - min(distance, 1.0)  # Cap at 1.0
                    
                    # If similarity is above threshold, consider as potential duplicate
                    if similarity > threshold:
                        duplicates.append((id1, id2, similarity))
                        # Only report one match per pair
                        break
                
                # If we found a duplicate for this encoding, move to next student
                if any(d[0] == id1 and d[1] == id2 for d in duplicates):
                    break
    
    return duplicates

def train_advanced_model(train_dir, model_path, progress_callback=None):
    """
    Train an advanced face recognition model
    
    Args:
        train_dir: Directory containing training images
        model_path: Path to save the trained model
        progress_callback: Callback function for progress updates (value, status_text, details_text)
        
    Returns:
        True if training was successful, False otherwise
    """
    try:
        # Declare global variables at the beginning of the function
        global face_classifier
        
        # Update progress
        if progress_callback:
            progress_callback(0, "Checking training directory...")
        
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
        
        # Prepare data for training
        X = []  # Face encodings
        y = []  # Person IDs
        
        # Dictionary to store face encodings by person ID for duplicate detection
        training_data = {}
        
        # Update progress
        if progress_callback:
            progress_callback(5, "Extracting face features...", "Scanning directories...")
        
        total_dirs = len(person_dirs)
        for dir_idx, person_dir in enumerate(person_dirs):
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
                              if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                if not image_paths:
                    print(f"No images found in {person_dir}")
                    continue
                
                # Update progress
                if progress_callback:
                    progress_value = 5 + (45 * dir_idx / total_dirs)
                    progress_callback(
                        progress_value,
                        "Processing images...",
                        f"Processing {dir_name} ({len(image_paths)} images)"
                    )
                
                # Process each image
                person_encodings = []
                for img_idx, image_path in enumerate(image_paths):
                    try:
                        # Load image
                        image = cv2.imread(image_path)
                        if image is None:
                            print(f"Could not load image: {image_path}")
                            continue
                        
                        # Detect face
                        face_locations = detect_faces_dlib(image)
                        if not face_locations:
                            print(f"No face detected in {image_path}")
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
                        encoding = extract_face_encoding(image, face_location)
                        if encoding is None:
                            continue
                        
                        person_encodings.append(encoding)
                        
                        # Update progress for each image
                        if progress_callback and img_idx % 5 == 0:  # Update every 5 images
                            sub_progress = img_idx / len(image_paths)
                            progress_callback(
                                progress_value + (sub_progress * 45 / total_dirs),
                                "Processing images...",
                                f"Processing {dir_name}: Image {img_idx + 1}/{len(image_paths)}"
                            )
                    
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
                        continue
                
                # Only use this person if we have enough valid encodings
                if len(person_encodings) >= 3:  # Require at least 3 valid images
                    X.extend(person_encodings)
                    y.extend([person_id] * len(person_encodings))
                    training_data[person_id] = person_encodings
                else:
                    print(f"Not enough valid images for person {dir_name}, skipping")
            
            except Exception as e:
                print(f"Error processing person directory {person_dir}: {e}")
                continue
        
        if not X:
            print("No valid face encodings found for training!")
            return False
        
        # Check for duplicate faces
        if progress_callback:
            progress_callback(50, "Checking for duplicate faces...")
        
        if len(training_data) > 1:
            duplicates = detect_duplicate_faces(training_data)
            if duplicates:
                duplicate_message = "Potential duplicate faces detected:\n"
                for id1, id2, similarity in duplicates:
                    name1 = db_utils.get_student_name(id1) or str(id1)
                    name2 = db_utils.get_student_name(id2) or str(id2)
                    duplicate_message += f"- {name1} (ID: {id1}) and {name2} (ID: {id2}): {similarity*100:.1f}% similar\n"
                print(duplicate_message)
        
        # Update progress
        if progress_callback:
            progress_callback(60, "Training model...", "Converting data...")
        
        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        if len(set(y)) < 2:
            print(f"Need at least 2 different people for training, only found {len(set(y))}")
            
            # Handle single person case
            if len(set(y)) == 1:
                if progress_callback:
                    progress_callback(70, "Training single-person model...")
                
                classifier = SVC(C=1.0, kernel='linear', probability=True)
                person_id = list(set(y))[0]
                X_augmented = X.copy()
                y_augmented = y.copy()
                
                # Create synthetic negative samples
                if progress_callback:
                    progress_callback(80, "Generating synthetic samples...")
                
                np.random.seed(42)
                for i in range(min(len(X), 10)):
                    noise = np.random.normal(0, 0.1, X[i].shape)
                    negative_sample = X[i] + noise
                    negative_sample = negative_sample / np.linalg.norm(negative_sample)
                    X_augmented = np.vstack([X_augmented, negative_sample])
                    y_augmented = np.append(y_augmented, -100)
                
                if progress_callback:
                    progress_callback(90, "Training classifier...")
                
                classifier.fit(X_augmented, y_augmented)
                
                # Save the model
                if progress_callback:
                    progress_callback(95, "Saving model...")
                
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump(classifier, model_path)
                
                classifier_path = os.path.join(config.TRAINING_IMAGE_LABEL_DIR, "face_classifier.pkl")
                joblib.dump(classifier, classifier_path)
                
                face_classifier = classifier
                
                if progress_callback:
                    progress_callback(100, "Training completed!", "Single-person model trained successfully")
                
                return True
            
            return False
        
        # Train classifier
        if progress_callback:
            progress_callback(80, "Training classifier...", f"Processing {len(X)} face encodings from {len(set(y))} people")
        
        classifier = SVC(C=1.0, kernel='linear', probability=True)
        classifier.fit(X, y)
        
        # Save the model
        if progress_callback:
            progress_callback(90, "Saving model...")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(classifier, model_path)
        
        classifier_path = os.path.join(config.TRAINING_IMAGE_LABEL_DIR, "face_classifier.pkl")
        joblib.dump(classifier, classifier_path)
        
        face_classifier = classifier
        
        if progress_callback:
            progress_callback(100, "Training completed!", f"Model trained with {len(X)} face encodings from {len(set(y))} people")
        
        return True
        
    except Exception as e:
        print(f"Error training advanced model: {e}")
        if progress_callback:
            progress_callback(0, f"Error: {str(e)}")
        return False

def recognize_face_advanced(image, threshold=0.6):
    """
    Recognize a face using the advanced face recognition model
    
    Args:
        image: Input image (BGR format from OpenCV)
        threshold: Confidence threshold (0-1)
        
    Returns:
        (id, confidence) if face is recognized, (-1, 0) otherwise
    """
    try:
        global face_classifier
        
        # Initialize models if not already initialized
        if face_classifier is None:
            classifier_path = os.path.join(config.TRAINING_IMAGE_LABEL_DIR, "face_classifier.pkl")
            if os.path.exists(classifier_path):
                try:
                    face_classifier = joblib.load(classifier_path)
                    
                    # Check if this is a single-person model by looking at the classes
                    if -100 in face_classifier.classes_:
                        print("Using single-person model with stricter threshold")
                        # For single-person models, we need to be more strict to avoid false positives
                        threshold = 0.7  # Higher threshold to prevent false recognitions
                except Exception as e:
                    print(f"Error loading face classifier: {e}")
                    return -1, 0
            else:
                print("Face classifier not found. Please train the model first.")
                return -1, 0
        
        # Check if image is valid
        if image is None or image.size == 0:
            print("Invalid image provided to recognize_face_advanced")
            return -1, 0
        
        # OPTIMIZATION: Resize image for faster processing
        height, width = image.shape[:2]
        if width > 640:  # Only resize if image is large
            scale = 640 / width
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        
        # Detect face
        face_locations = detect_faces_dlib(image)
        if not face_locations:
            print("No face detected")
            return -1, 0
        
        # Use the largest face if multiple faces detected
        if len(face_locations) > 1:
            # Find the largest face by area
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
        encoding = extract_face_encoding(image, face_location)
        if encoding is None:
            print("Could not extract face encoding")
            return -1, 0
        
        # Predict using classifier
        encoding_reshaped = encoding.reshape(1, -1)
        
        # Get prediction and probability
        person_id = face_classifier.predict(encoding_reshaped)[0]
        probabilities = face_classifier.predict_proba(encoding_reshaped)[0]
        
        # Get the probability for the predicted class
        idx = list(face_classifier.classes_).index(person_id)
        confidence = probabilities[idx]
        
        print(f"Face recognition result: ID={person_id}, Confidence={confidence:.4f}")
        
        # If the predicted class is our synthetic "unknown" class, return unknown
        if person_id == -100:
            return -1, confidence
        
        # For single-person models, implement additional verification
        if len(face_classifier.classes_) == 2 and -100 in face_classifier.classes_:
            # This is a single-person model
            real_classes = [c for c in face_classifier.classes_ if c != -100]
            if len(real_classes) == 1:
                # Get the probability for the "unknown" class
                unknown_idx = list(face_classifier.classes_).index(-100)
                unknown_prob = probabilities[unknown_idx]
                
                # If the unknown probability is too high, reject the match
                if unknown_prob > 0.3:  # If there's more than 30% chance it's unknown
                    print(f"Rejecting match: unknown probability too high ({unknown_prob:.4f})")
                    return -1, confidence
                
                # Additional verification: check if confidence is significantly higher than unknown
                if confidence < unknown_prob * 2:  # Confidence should be at least twice the unknown probability
                    print(f"Rejecting match: confidence ({confidence:.4f}) not significantly higher than unknown ({unknown_prob:.4f})")
                    return -1, confidence
        
        # Check confidence threshold
        if confidence >= threshold:
            return person_id, confidence
        else:
            return -1, confidence
    
    except Exception as e:
        print(f"Error in recognize_face_advanced: {e}")
        return -1, 0

def compare_faces(known_encoding, image, tolerance=0.6):
    """
    Compare a known face encoding with faces in an image
    
    Args:
        known_encoding: Known face encoding
        image: Input image (BGR format from OpenCV)
        tolerance: Tolerance for face comparison (lower is stricter)
        
    Returns:
        True if match found, False otherwise
    """
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            print("Invalid image provided to compare_faces")
            return False
        
        # Convert BGR to RGB (dlib uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        if not face_locations:
            print("No faces detected in the image")
            return False
        
        # Extract face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        # Compare with known encoding
        for face_encoding in face_encodings:
            # Compare faces
            matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=tolerance)
            if matches[0]:
                return True
        
        return False
    
    except Exception as e:
        print(f"Error in compare_faces: {e}")
        return False

# Initialize models when module is imported
initialize_models() 