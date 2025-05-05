# Indian Sign Language Detection

This project provides a web-based application for detecting and translating Indian Sign Language gestures into text and speech.

## Features

- **Gesture to Text**: Convert hand gestures to text in real-time
- **Gesture to Speech**: Convert hand gestures to spoken words
- **Speech to Gesture**: Convert spoken words to animated sign language gestures
- **Voice Translation**: Translate between different languages

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Flask
- OpenCV
- MediaPipe
- NumPy

### Installation

1. Clone or download this repository
2. Install the required Python packages:

```bash
pip install flask opencv-python mediapipe numpy
```

### Running the Application

1. Navigate to the project directory
2. Start the Flask server:

```bash
cd hand-gesture-recognition-mediapipe-main
python app.py
```

3. Open your web browser and go to `http://localhost:5000/`

## Usage

1. From the home page, click on "Gesture to Text"
2. Follow the instructions to start the Flask server
3. Once the server is running, click the "Open Gesture Recognition App" button
4. Position your hand in front of the camera to detect gestures
5. The detected gestures will be displayed on the screen

## Project Structure

- `index.html`: Home page of the application
- `gesture-to-text.html`: Instructions for starting the gesture recognition app
- `hand-gesture-recognition-mediapipe-main/`: Directory containing the Flask app and gesture recognition code
  - `app.py`: Main Flask application
  - `templates/`: HTML templates
  - `static/`: Static files (CSS, JavaScript)
  - `model/`: Trained models for gesture recognition
  - `utils/`: Utility functions

## License

This project is licensed under the MIT License - see the LICENSE file for details.
