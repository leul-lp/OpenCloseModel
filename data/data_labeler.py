import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
# Set max_num_hands=1 to ensure we only get landmarks for a single hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Setup webcam
cap = cv2.VideoCapture(0)

# Create a CSV file to store the landmarks
csv_file_path = 'hand_gesture_data_2.csv'
csv_file = open(csv_file_path, 'w', newline='')
csv_writer = csv.writer(csv_file)

# Create the column headers for the CSV file (21 landmarks * 3 coords + 1 label = 64 fields)
headers = []
for i in range(21):
    headers.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
headers.append('label')
csv_writer.writerow(headers)
print(f"CSV file '{csv_file_path}' created with headers.")

# Main loop to capture and process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: # If the status is not False [True] break 
        break

    # Flip the frame horizontally for a more intuitive view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    feedback_text = ""
    landmark_data = [] # Initialize landmark data list for each frame

    # Check if a hand was detected
    if results.multi_hand_landmarks:
        # We only expect one hand due to max_num_hands=1
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw the landmarks on the frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract the coordinates and flatten them into a list
        for landmark in hand_landmarks.landmark: # 21 landmarks
            landmark_data.extend([landmark.x, landmark.y, landmark.z]) # 3 values total -> 63 values appended values per frame 

    # Check for keyboard input to save data
    key = cv2.waitKey(1) & 0xFF
    
    # Check if a hand was detected before saving
    if landmark_data:
        # Press '0' to save as 'fist'
        if key == ord('0'):
            landmark_data.append('fist') # Adds it at the end
            csv_writer.writerow(landmark_data)
            feedback_text = "Fist gesture saved!"
            print("Fist gesture saved!")

        # Press '1' to save as 'open_palm'
        elif key == ord('1'):
            landmark_data.append('open_palm') # Adds it at the end
            csv_writer.writerow(landmark_data)
            feedback_text = "Open palm gesture saved!"
            print("Open palm gesture saved!")

    # Display feedback text on the screen
    cv2.putText(frame, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Hand Gesture Data Collector', frame)
    
    # Exit on 'q' press
    if key == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
csv_file.close()

