import mediapipe as mp
import cv2 as cv
import time as t
import sys 

import torch.nn as nn
import torch

from data.model.OCNET_arch import OCNet # Import the model architecture
import joblib


scaler = joblib.load('./data/model/scaler.pkl')

capture = cv.VideoCapture(0)

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

hands = mpHands.Hands()

# Initialize the model
new_model = OCNet()

new_model.load_state_dict(torch.load('./data/model/ocnet_model_weight.pth'))

new_model.eval()
print("Model Save Loaded Sucessfully")

def predict(landmarks):
    """
    Takes a MediaPipe NormalizedLandmarkList, preprocesses it, and returns a prediction.
    """
    # 1. Convert the landmarks object into a flat list of numbers
    landmarks_list = []
    # Check if landmarks are detected
    if landmarks:
        # Loop through each landmark and append its x, y, and z coordinates
        for lm in landmarks.landmark:
            landmarks_list.append(lm.x)
            landmarks_list.append(lm.y)
            landmarks_list.append(lm.z)


    landmarks_tensor = torch.tensor(landmarks_list, dtype=torch.float32).reshape(1, -1)
    
    try:
        new_data_scaled = scaler.transform(landmarks_tensor.numpy())
    except Exception as e:
        print(f"Error during scaling: {e}")
        return None


    tensor_data = torch.from_numpy(new_data_scaled).type(torch.float32)
    

    with torch.inference_mode():
        output = new_model(tensor_data).squeeze()
        probs = torch.sigmoid(output)
        predicted_class = torch.round(probs)
    
    return predicted_class.item() # Return the predicted value



while True:
    _, frame = capture.read()

    frame = cv.flip(frame, 1)

    imgConvt = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgConvt)

    land_marks = results.multi_hand_landmarks
    
    if land_marks and results.multi_handedness:
        for land_mark, hand_handedness in zip(land_marks, results.multi_handedness):
            mpDraw.draw_landmarks(
                frame, land_mark, 
                mpHands.HAND_CONNECTIONS, 
                landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 0), thickness=5, circle_radius=8),
                connection_drawing_spec=mpDraw.DrawingSpec(color=(255, 255, 255), thickness=5)
                )
            

            # For drawing the box around the hand 
            h, w, c = frame.shape

            x_min, y_min = w, h
            x_max, y_max = 0, 0

            padding = 10

            for lm in land_mark.landmark:
                x, y = int(lm.x * w), int(lm.y * h)

                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y

            cv.rectangle(frame, (x_min-padding, y_min-padding), (x_max+padding, y_max+padding),(100, 0, 255), 5)
            
            hand_type = hand_handedness.classification[0].label
            value = predict(land_mark)
            prediction = "palm" if value else "fist" 
            cv.putText(frame, f"Prediction: {prediction}({hand_type})", (x_min - padding, y_max + padding + 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            
    cv.imshow("Preview", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)