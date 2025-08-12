# Hand Gesture Predictor - Open/Close Detection

## Overview
This project is a real-time hand gesture recognition system that can detect and classify hand gestures as either "palm" (open hand) or "fist" (closed hand) using computer vision and deep learning. The system uses a webcam to capture hand movements, MediaPipe for hand landmark detection, and a custom neural network (OCNet) for gesture classification.

## Features
- Real-time hand detection and tracking
- Classification of hand gestures (palm/fist)
- Visual feedback with hand landmarks and bounding box
- Support for both left and right hands
- Pre-trained model for immediate use

## Project Structure
```
├── README.md               # Project documentation
├── main.py                 # Main application script
├── data/                   # Data directory
│   ├── analyses.py         # Data analysis scripts
│   ├── data_labeler.py     # Tool for collecting and labeling hand gesture data
│   ├── hand_gesture_data.csv # Training dataset
│   └── model/              # Model directory
│       ├── OCNET_arch.py   # Neural network architecture
│       ├── model_train.ipynb # Jupyter notebook for model training
│       ├── ocnet_model_weight.pth # Pre-trained model weights
│       └── scaler.pkl      # Data standardization scaler
```

## Technologies Used
- **Python**: Core programming language
- **OpenCV**: For webcam access and image processing
- **MediaPipe**: For hand landmark detection
- **PyTorch**: For building and training the neural network
- **Scikit-learn**: For data preprocessing and evaluation
- **Pandas**: For data manipulation
- **Matplotlib**: For data visualization

## How It Works
1. **Hand Detection**: The system uses MediaPipe's hand detection to identify hands in the webcam feed
2. **Landmark Extraction**: 21 hand landmarks (63 coordinates) are extracted from the detected hand
3. **Preprocessing**: The landmark data is standardized using a pre-trained scaler
4. **Classification**: The processed data is fed into the OCNet neural network
5. **Prediction**: The model outputs a prediction (palm or fist) with visual feedback

## Model Architecture
The OCNet (Open-Close Network) is a simple feedforward neural network with the following architecture:
- Input layer: 63 neurons (21 landmarks × 3 coordinates)
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 80 neurons with ReLU activation
- Hidden layer 3: 50 neurons with ReLU activation
- Output layer: 1 neuron (sigmoid activation for binary classification)
- Dropout layers (0.2) for regularization

## Setup and Installation

### Prerequisites
- Python 3.7+
- Webcam

### Installation
1. Clone the repository
2. Install the required packages:
   ```
   pip install opencv-python mediapipe torch torchvision pandas matplotlib scikit-learn joblib
   ```
3. Run the application:
   ```
   python main.py
   ```

## Usage
- Run the main.py script to start the application
- Position your hand in front of the webcam
- The system will detect your hand and classify the gesture as either "palm" or "fist"
- Press 'q' to exit the application

## Data Collection
The project includes a data collection tool (`data_labeler.py`) that allows you to create your own dataset:
1. Run the data labeler script
2. Show hand gestures to the camera
3. Press '0' to save a fist gesture or '1' to save an open palm gesture
4. Press 'q' to exit

## Training Your Own Model
You can train your own model using the provided Jupyter notebook (`model_train.ipynb`):
1. Collect data using the data labeler
2. Open the notebook and adjust parameters as needed
3. Run the training cells
4. Save your model weights and scaler

## Future Improvements
- Add support for more gesture types
- Implement transfer learning for better accuracy
- Create a graphical user interface
- Add gesture-controlled applications

## License
This project is open-source and available for personal and educational use.

## Acknowledgments
- MediaPipe team for the hand tracking solution
- PyTorch community for the deep learning framework
# OpenCloseModel
# OpenCloseModel
