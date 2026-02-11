from utils.webcam import WebcamCapture

def main():
    """Webcam demonstration for face capture."""
    print("NGO Facial Image Analysis - Webcam Demo")
    print("Press 'c' to capture face, 'q' to quit")
    
    try:
        webcam = WebcamCapture()
        webcam.initialize()
        
        print("Starting webcam...")
        success = webcam.capture_and_save_face("captured_face.jpg")
        
        if success:
            print("Face captured successfully!")
            print("File saved as: captured_face.jpg")
        else:
            print("Face capture cancelled")
            
        webcam.close()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your webcam is connected and working")

if __name__ == "__main__":
    main()