import cv2
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class FaceDetector:
    """
    Face detection module using OpenCV's DNN module with pre-trained Caffe model.
    Designed for ethical NGO use - detects faces but doesn't identify individuals.
    """
    
    def __init__(self):
        # Try to load DNN model first, fallback to Haar cascade
        try:
            # Try DNN model
            self.net = cv2.dnn.readNetFromCaffe(
                "deploy.prototxt.txt",
                "res10_300x300_ssd_iter_140000.caffemodel"
            )
            self.use_dnn = True
        except:
            # Fallback to Haar cascade
            self.face_cascade = cv2.CascadeClassifier()
            self.face_cascade.load(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.use_dnn = False
        
        self.confidence_threshold = 0.5
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of bounding boxes (x, y, w, h) for detected faces
        """
        if self.use_dnn:
            return self._detect_faces_dnn(image)
        else:
            return self._detect_faces_haar(image)
    
    def _detect_faces_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using DNN model.
        """
        # Preprocess image for face detection
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        # Perform face detection
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter weak detections
            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are within image bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Calculate width and height
                face_box = (startX, startY, endX - startX, endY - startY)
                faces.append(face_box)
        
        return faces
    
    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Haar cascade (fallback method).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Convert to list of tuples
        faces_list = []
        for (x, y, w, h) in faces:
            faces_list.append((x, y, w, h))
        
        return faces_list
    
    def visualize_biometric_capture(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Visualize what biometric features are being captured from detected faces.
        
        Args:
            image: Original input image
            faces: List of detected face bounding boxes
            
        Returns:
            Image showing the captured biometric regions
        """
        output_image = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract the face region that will be used for biometric processing
            face_roi = image[y:y+h, x:x+w]
            
            # Resize to standard recognition size (if needed for embedding)
            # This shows what the recognition system actually "sees"
            processed_face = cv2.resize(face_roi, (160, 160))
            
            # Draw original detection box (green)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_image, f"Face {i+1}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw processed face region box (blue)
            # Show where the biometric features are actually extracted from
            processed_x = x + w + 20
            processed_y = y
            cv2.rectangle(output_image, (processed_x, processed_y), 
                         (processed_x + 160, processed_y + 160), (255, 0, 0), 2)
            
            # Place the processed face image as overlay
            # This shows exactly what the recognition system uses
            try:
                # Calculate position for overlay
                overlay_x = processed_x
                overlay_y = processed_y
                
                # Place the processed face image
                output_image[overlay_y:overlay_y+160, overlay_x:overlay_x+160] = processed_face
                
                # Add label for processed region
                cv2.putText(output_image, "Processed", (processed_x, processed_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(output_image, "for Recognition", (processed_x, processed_y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except:
                # If overlay fails, just draw the box
                pass
        
        return output_image
    
    def draw_detections(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw bounding boxes on detected faces.
        
        Args:
            image: Input image
            faces: List of face bounding boxes
            
        Returns:
            Image with bounding boxes drawn
        """
        output_image = image.copy()
        
        for (x, y, w, h) in faces:
            # Draw green rectangle around face
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label with confidence (placeholder)
            cv2.putText(output_image, "Face", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output_image

class FaceDetectorInterface:
    """
    Interface for visualizing face detection and biometric capture process.
    """
    
    def __init__(self):
        self.detector = FaceDetector()
    
    def show_biometric_capture(self, image_path: str):
        """
        Show the complete biometric capture process visualization.
        
        Args:
            image_path: Path to the image to process
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image from {image_path}")
            return
        
        # Detect faces
        faces = self.detector.detect_faces(image)
        print(f"Detected {len(faces)} faces")
        
        if len(faces) == 0:
            print("No faces detected")
            return
        
        # Create visualization
        visualization = self.detector.visualize_biometric_capture(image, faces)
        
        # Display the result
        cv2.imshow("Biometric Capture Visualization", visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("Visualization complete!")
        
        # Also save the visualization
        output_path = image_path.replace(".", "_biometric_visualization.")
        cv2.imwrite(output_path, visualization)
        print(f"Visualization saved to: {output_path}")

class CompleteVisualizationInterface:
    """
    Complete interface showing the entire biometric recognition pipeline.
    """
    
    def __init__(self):
        self.detector_interface = FaceDetectorInterface()
    
    def show_complete_pipeline(self, image_path: str):
        """
        Show the complete biometric recognition pipeline visualization.
        
        Args:
            image_path: Path to the image to process
        """
        print("Starting complete biometric recognition pipeline visualization...")
        print("=" * 50)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image from {image_path}")
            return
        
        print(f"Loaded image: {image.shape}")
        print()
        
        # Detect faces
        detector = FaceDetector()
        faces = detector.detect_faces(image)
        print(f"Detected {len(faces)} faces")
        
        if len(faces) == 0:
            print("No faces detected")
            return
        
        print("Face detection complete!")
        print()
        
        # Show biometric capture visualization
        print("Creating biometric capture visualization...")
        biometric_visualization = detector.visualize_biometric_capture(image, faces)
        
        # Save and show biometric visualization
        biometric_output_path = image_path.replace(".", "_biometric_visualization.")
        cv2.imwrite(biometric_output_path, biometric_visualization)
        print(f"Biometric visualization saved to: {biometric_output_path}")
        print()
        
        # Show step-by-step breakdown
        print("Creating step-by-step breakdown...")
        breakdown_images = []
        
        for i, (x, y, w, h) in enumerate(faces):
            print(f"Processing face {i+1}...")
            
            # Original face region
            face_roi = image[y:y+h, x:x+w]
            
            # Processed face region
            processed_face = cv2.resize(face_roi, (160, 160))
            
            # Create breakdown image
            breakdown_height = max(h, 160) + 60
            breakdown_width = w + 180 + 160
            breakdown_image = np.zeros((breakdown_height, breakdown_width, 3), dtype=np.uint8)
            breakdown_image.fill(240)  # Light gray background
            
            # Original face
            breakdown_image[30:30+h, 30:30+w] = face_roi
            cv2.rectangle(breakdown_image, (30, 30), (30+w, 30+h), (0, 255, 0), 2)
            cv2.putText(breakdown_image, "Original Face", (30, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Processed face
            breakdown_image[30:30+160, 30+w+60:30+w+60+160] = processed_face
            cv2.rectangle(breakdown_image, (30+w+60, 30), (30+w+60+160, 30+160), (255, 0, 0), 2)
            cv2.putText(breakdown_image, "Processed for", (30+w+60, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(breakdown_image, "Recognition", (30+w+60, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Add labels
            cv2.putText(breakdown_image, f"Face {i+1}", (10, breakdown_height-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            breakdown_images.append(breakdown_image)
        
        print("Step-by-step breakdown complete!")
        print()
        
        # Create final composite
        print("Creating final composite visualization...")
        
        # Combine all visualizations
        final_composite = np.vstack([
            np.hstack([image, biometric_visualization]),
            *breakdown_images
        ])
        
        # Save final composite
        composite_output_path = image_path.replace(".", "_complete_pipeline.")
        cv2.imwrite(composite_output_path, final_composite)
        print(f"Complete pipeline visualization saved to: {composite_output_path}")
        print()
        
        # Display results
        print("Displaying visualizations...")
        cv2.imshow("Biometric Capture Visualization", biometric_visualization)
        cv2.imshow("Complete Pipeline Visualization", final_composite)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("=" * 50)
        print("Complete biometric recognition pipeline visualization complete!")
        print("Visualization shows:")
        print("1. Original image with face detection")
        print("2. Biometric capture process (green = original, blue = processed)")
        print("3. Step-by-step breakdown of face processing")
        print("4. Final composite showing complete pipeline")
        print("=" * 50)

if __name__ == "__main__":
    # Test with Kanye West image
    interface = CompleteVisualizationInterface()
    image_path = "face_recognition_npo/test_images/kanye_west.jpeg"
    interface.show_complete_pipeline(image_path)

class FaceDetector:
    """
    Face detection module using OpenCV's DNN module with pre-trained Caffe model.
    Designed for ethical NGO use - detects faces but doesn't identify individuals.
    """
    
    def __init__(self):
        # Try to load DNN model first, fallback to Haar cascade
        try:
            # Try DNN model
            self.net = cv2.dnn.readNetFromCaffe(
                "deploy.prototxt.txt",
                "res10_300x300_ssd_iter_140000.caffemodel"
            )
            self.use_dnn = True
        except:
            # Fallback to Haar cascade
            self.face_cascade = cv2.CascadeClassifier()
            self.face_cascade.load(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.use_dnn = False
        
        self.confidence_threshold = 0.5
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of bounding boxes (x, y, w, h) for detected faces
        """
        if self.use_dnn:
            return self._detect_faces_dnn(image)
        else:
            return self._detect_faces_haar(image)
    
    def _detect_faces_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using DNN model.
        """
        # Preprocess image for face detection
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        # Perform face detection
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter weak detections
            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are within image bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Calculate width and height
                face_box = (startX, startY, endX - startX, endY - startY)
                faces.append(face_box)
        
        return faces
    
    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Haar cascade (fallback method).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Convert to list of tuples
        faces_list = []
        for (x, y, w, h) in faces:
            faces_list.append((x, y, w, h))
        
        return faces_list
    
    def visualize_biometric_capture(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Visualize what biometric features are being captured from detected faces.
        
        Args:
            image: Original input image
            faces: List of detected face bounding boxes
            
        Returns:
            Image showing the captured biometric regions
        """
        output_image = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract the face region that will be used for biometric processing
            face_roi = image[y:y+h, x:x+w]
            
            # Resize to standard recognition size (if needed for embedding)
            # This shows what the recognition system actually "sees"
            processed_face = cv2.resize(face_roi, (160, 160))
            
            # Draw original detection box (green)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_image, f"Face {i+1}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw processed face region box (blue)
            # Show where the biometric features are actually extracted from
            processed_x = x + w + 20
            processed_y = y
            cv2.rectangle(output_image, (processed_x, processed_y), 
                         (processed_x + 160, processed_y + 160), (255, 0, 0), 2)
            
            # Place the processed face image as overlay
            # This shows exactly what the recognition system uses
            try:
                # Calculate position for overlay
                overlay_x = processed_x
                overlay_y = processed_y
                
                # Place the processed face image
                output_image[overlay_y:overlay_y+160, overlay_x:overlay_x+160] = processed_face
                
                # Add label for processed region
                cv2.putText(output_image, "Processed", (processed_x, processed_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(output_image, "for Recognition", (processed_x, processed_y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except:
                # If overlay fails, just draw the box
                pass
        
        return output_image
    
    def draw_detections(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw bounding boxes on detected faces.
        
        Args:
            image: Input image
            faces: List of face bounding boxes
            
        Returns:
            Image with bounding boxes drawn
        """
        output_image = image.copy()
        
        for (x, y, w, h) in faces:
            # Draw green rectangle around face
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label with confidence (placeholder)
            cv2.putText(output_image, "Face", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output_image

class FaceDetectorInterface:
    """
    Interface for visualizing face detection and biometric capture process.
    """
    
    def __init__(self):
        self.detector = FaceDetector()
    
    def show_biometric_capture(self, image_path: str):
        """
        Show the complete biometric capture process visualization.
        
        Args:
            image_path: Path to the image to process
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image from {image_path}")
            return
        
        # Detect faces
        faces = self.detector.detect_faces(image)
        print(f"Detected {len(faces)} faces")
        
        if len(faces) == 0:
            print("No faces detected")
            return
        
        # Create visualization
        visualization = self.detector.visualize_biometric_capture(image, faces)
        
        # Display the result
        cv2.imshow("Biometric Capture Visualization", visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("Visualization complete!")
        
        # Also save the visualization
        output_path = image_path.replace(".", "_biometric_visualization.")
        cv2.imwrite(output_path, visualization)
        print(f"Visualization saved to: {output_path}")

class FaceDetector:
    """
    Face detection module using OpenCV's DNN module with pre-trained Caffe model.
    Designed for ethical NGO use - detects faces but doesn't identify individuals.
    """
    
    def __init__(self):
        # Load pre-trained face detection model
        self.net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt.txt",
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        self.confidence_threshold = 0.5
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of bounding boxes (x, y, w, h) for detected faces
        """
        # Preprocess image for face detection
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        # Perform face detection
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter weak detections
            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are within image bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Calculate width and height
                face_box = (startX, startY, endX - startX, endY - startY)
                faces.append(face_box)
        
        return faces
    
    def visualize_biometric_capture(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Visualize what biometric features are being captured from detected faces.
        
        Args:
            image: Original input image
            faces: List of detected face bounding boxes
            
        Returns:
            Image showing the captured biometric regions
        """
        output_image = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract the face region that will be used for biometric processing
            face_roi = image[y:y+h, x:x+w]
            
            # Resize to standard recognition size (if needed for embedding)
            # This shows what the recognition system actually "sees"
            processed_face = cv2.resize(face_roi, (160, 160))
            
            # Draw original detection box (green)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_image, f"Face {i+1}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw processed face region box (blue)
            # Show where the biometric features are actually extracted from
            processed_x = x + w + 20
            processed_y = y
            cv2.rectangle(output_image, (processed_x, processed_y), 
                         (processed_x + 160, processed_y + 160), (255, 0, 0), 2)
            
            # Place the processed face image as overlay
            # This shows exactly what the recognition system uses
            try:
                # Calculate position for overlay
                overlay_x = processed_x
                overlay_y = processed_y
                
                # Place the processed face image
                output_image[overlay_y:overlay_y+160, overlay_x:overlay_x+160] = processed_face
                
                # Add label for processed region
                cv2.putText(output_image, "Processed", (processed_x, processed_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(output_image, "for Recognition", (processed_x, processed_y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except:
                # If overlay fails, just draw the box
                pass
        
        return output_image


def load_face_detection_model():
    """
    Load face detection model weights if not already present.
    """
    # Model files should be downloaded separately due to size
    # For now, assume they're in the correct location
    pass

if __name__ == "__main__":
    # Test face detection with a sample image
    detector = FaceDetector()
    
    # Load test image (you'll need to provide one)
    # image = cv2.imread("test_image.jpg")
    # faces = detector.detect_faces(image)
    # print(f"Detected {len(faces)} faces")
    # 
    # result_image = detector.draw_detections(image, faces)
    # cv2.imshow("Face Detection Result", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    print("Face detection module loaded successfully")