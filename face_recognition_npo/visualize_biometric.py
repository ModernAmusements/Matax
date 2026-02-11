#!/usr/bin/env python3
"""
Simple biometric visualization script.
Shows what biometric features are captured for facial recognition.

Usage:
    python visualize_biometric.py                    # Uses default test image
    TEST_IMAGE=my_image.jpg python visualize_biometric.py  # Uses custom image
"""

import os
import cv2
import numpy as np

# Configuration - set via environment variable or use default
DEFAULT_IMAGE = 'test_images/kanye_west.jpeg'
IMAGE_PATH = os.environ.get('TEST_IMAGE', DEFAULT_IMAGE)


def visualize_biometric_capture(image_path: str = IMAGE_PATH):
    """
    Visualize what biometric features are being captured from detected faces.
    
    Args:
        image_path: Path to the image to process
    """
    print("Starting biometric capture visualization...")
    print("=" * 50)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return
    
    print(f"Loaded image: {image.shape}")
    
    # Use Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    print(f"Detected {len(faces)} faces")
    
    if len(faces) == 0:
        print("No faces detected")
        return
    
    print("Face detection complete!")
    print()
    
    # Create visualization
    output_image = image.copy()
    
    for i, (x, y, w, h) in enumerate(faces):
        print(f"Processing face {i+1}...")
        
        # Extract the face region
        face_roi = image[y:y+h, x:x+w]
        
        # Resize to standard recognition size (160x160)
        processed_face = cv2.resize(face_roi, (160, 160))
        
        # Draw original detection box (green)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_image, f"Face {i+1}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw processed face region box (blue)
        processed_x = x + w + 20
        processed_y = y
        cv2.rectangle(output_image, (processed_x, processed_y), 
                     (processed_x + 160, processed_y + 160), (255, 0, 0), 2)
        
        # Place the processed face image as overlay
        try:
            output_image[processed_y:processed_y+160, processed_x:processed_x+160] = processed_face
            cv2.putText(output_image, "Processed", (processed_x, processed_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(output_image, "for Recognition", (processed_x, processed_y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except:
            pass
    
    print()
    print("Creating visualization...")
    
    # Save visualization
    output_path = image_path.replace(".", "_biometric_visualization.")
    cv2.imwrite(output_path, output_image)
    print(f"Biometric visualization saved to: {output_path}")
    print()
    
    # Display results
    print("Displaying visualization...")
    cv2.imshow("Biometric Capture Visualization", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("=" * 50)
    print("Biometric capture visualization complete!")
    print()
    print("Legend:")
    print("🟩 GREEN = Original detected face region")
    print("🔵 BLUE = Processed region (160x160) used for biometric features")
    print("=" * 50)

if __name__ == "__main__":
    visualize_biometric_capture(IMAGE_PATH)
