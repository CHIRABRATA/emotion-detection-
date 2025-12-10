import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import av
from tensorflow.keras.models import load_model



@st.cache_resource
def load_emotion_model():
    model = load_model("emotion_mobilenet_fast (3).keras")
    return model

try:
    model = load_emotion_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]

            # Preprocess exactly like training
            final_image = cv2.resize(roi, (128, 128))
            final_image = final_image.astype("float32") / 255.0
            final_image = np.expand_dims(final_image, axis=0)

            # Predict
            prediction = model.predict(final_image)
            max_index = np.argmax(prediction)
            emotion = EMOTION_LABELS[max_index]
            confidence = np.max(prediction)

            # Draw results
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{emotion} ({int(confidence * 100)}%)",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------------------------------------------------------
# 3. Streamlit UI
# -----------------------------------------------------------------------------
st.title("📸 Live Emotion Detector")
st.write("Real-time facial emotion recognition using MobileNetV2")

webrtc_streamer(
    key="emotion-detector",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True  # 🔥 Important fix
)
