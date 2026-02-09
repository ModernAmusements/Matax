import cv2
import numpy as np
from typing import Optional
import time

class WebcamCapture:
    """
    Webcam capture functionality for testing and demonstration.
    Allows real-time face detection and embedding extraction.
    """
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.face_detector = None
        self.embedding_extractor = None
        self.comparator = None
        
    def initialize(self):
        """
        Initialize webcam and all required components.
        """
        # Open webcam
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open camera with index {self.camera_index}")
        
        # Initialize face detection
        self.face_detector = FaceDetector()
        
        # Initialize embedding extraction
        self.embedding_extractor = FaceNetEmbeddingExtractor()
        
        # Initialize similarity comparison
        self.comparator = SimilarityComparator()
        
        print(f"Webcam initialized with resolution: {self.cap.get(3)}x{self.cap.get(4)}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from webcam.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a captured frame.
        """
        return self.face_detector.detect_faces(frame)
    
    def extract_embedding_from_face(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding from a detected face.
        """
        return self.embedding_extractor.extract_embedding(face_image)
    
    def process_live_video(self, 
                          reference_embeddings: List[np.ndarray],
                          reference_ids: List[str],
                          duration: int = 30):
        """
        Process live video stream with real-time face detection and comparison.
        
        Args:
            reference_embeddings: List of reference embeddings for comparison
            reference_ids: Corresponding reference IDs
            duration: Duration to run in seconds
        """
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Capture frame
            frame = self.capture_frame()
            if frame is None:
                break
                
            # Detect faces
            faces = self.detect_faces_in_frame(frame)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_image = frame[y:y+h, x:x+w]
                
                # Extract embedding
                embedding = self.extract_embedding_from_face(face_image)
                
                if embedding is not None:
                    # Compare with reference embeddings
                    results = self.comparator.compare_embeddings(
                        embedding, reference_embeddings, reference_ids
                    )
                    
                    # Display results
                    self._display_results(frame, (x, y, w, h), results)
            
            # Show frame
            cv2.imshow("Live Face Analysis", frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _display_results(self, frame: np.ndarray, face_box: Tuple[int, int, int, int],
                        results: List[Tuple[str, float]]):
        """
        Display comparison results on the frame.
        """
        x, y, w, h = face_box
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display top results
        for i, (ref_id, similarity) in enumerate(results[:3]):
            confidence = self.comparator.get_confidence_band(similarity)
            text = f"{ref_id}: {similarity:.2f} ({confidence})"
            
            # Position text above face
            text_y = y - 15 - (i * 20)
            if text_y < 0:
                text_y = y + h + 15 + (i * 20)
                
            cv2.putText(frame, text, (x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def capture_and_save_face(self, output_path: str) -> bool:
        """
        Capture a face from webcam and save it.
        """
        print("Press 'c' to capture face, 'q' to quit")
        
        while True:
            frame = self.capture_frame()
            if frame is None:
                break
                
            # Detect faces
            faces = self.detect_faces_in_frame(frame)
            
            # Draw detections
            frame_with_boxes = self.face_detector.draw_detections(frame, faces)
            
            cv2.imshow("Capture Face", frame_with_boxes)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and faces:
                # Capture first detected face
                x, y, w, h = faces[0]
                face_image = frame[y:y+h, x:x+w]
                
                # Save face image
                cv2.imwrite(output_path, face_image)
                print(f"Face saved to {output_path}")
                cv2.destroyAllWindows()
                return True
            
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False
    
    def close(self):
        """
        Release webcam and clean up resources.
        """
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test webcam functionality
    webcam = WebcamCapture()
    
    try:
        webcam.initialize()
        print("Webcam capture module initialized successfully")
        
        # Test face capture
        # webcam.capture_and_save_face("test_face.jpg")
        
    except Exception as e:
        print(f"Error initializing webcam: {e}")
    finally:
        webcam.close()