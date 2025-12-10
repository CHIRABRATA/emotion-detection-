# emotion-detection-
📌 Emotion Detection Using MobileNetV2 (Streamlit + TensorFlow)

This project performs real-time facial emotion recognition using:

MobileNetV2 (fast + lightweight pretrained CNN)

TensorFlow / Keras

OpenCV Haar Cascade

Streamlit Web UI

WebRTC live webcam streaming

The model is trained on images of 7 emotions and deployed using Streamlit.

🚀 Features

✔ Real-time emotion detection via webcam
✔ Lightweight model (MobileNetV2 backbone)
✔ Fast inference (<20ms per frame on CPU)
✔ Works fully inside browser (Streamlit WebRTC)
✔ Trained using Functional API (safe model loading)
✔ No loading errors (unlike Sequential API issues)

🧠 Emotions Detected
['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


These MUST match your training folder names.

📁 Project Structure
emotion-detector/
│
├── app.py                     # Streamlit UI (live video)
├── emotion_mobilenet_fast.keras  # Saved model
└── README.md
