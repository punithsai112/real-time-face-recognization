import cv2
import mediapipe as mp
# Load the image
import matplotlib as plt
image = cv2.imread(r"C:\Users\unkno\Downloads\kk.jpg")
cv2.imshow("image",image)
# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Convert image color for mediapipe (BGR to RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(image_rgb)

    # Draw face detections
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)

# Show the result
cv2.imshow("Mediapipe Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
