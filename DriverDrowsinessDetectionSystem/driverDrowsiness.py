import cv2
import dlib
from scipy.spatial import distance
import threading
import pygame

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")

# Define EAR threshold and consecutive frame count threshold
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

# Initialize counters
frame_count = 0

# Calculate the Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])  
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to play the alert sound in a separate thread

pygame.mixer.init()

def play_alert_sound():
    pygame.mixer.music.load("alert.wav")
    pygame.mixer.music.play()

# Main loop for drowsiness detection
cap = cv2.VideoCapture(0)  # Use 0 for the laptop's internal webcam

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        # Extract coordinates for left and right eyes
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        
        # Calculate EAR for both eyes and take the average
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Check if EAR is below the threshold
        if ear < EAR_THRESHOLD:
            frame_count += 1
            if frame_count >= CONSEC_FRAMES:
                cv2.putText(frame, "Drowsiness Alert!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Play alert sound in a new thread to avoid blocking
                threading.Thread(target=play_alert_sound).start()
        else:
            frame_count = 0
        # Draw landmarks for visual feedback
        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
    
    # Display the frame
    cv2.imshow("Driver Drowsiness Detection", frame)
    
    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
