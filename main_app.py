import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import pygame
import threading

# load model
emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

pygame.mixer.init()

# Load sound
alert_sound = pygame.mixer.Sound("bip_sound.mp3")


# Function to play alert sound
def play_alert_sound():
    alert_sound.play()


# load weights into new model
classifier.load_weights("emotion_model1.h5")

# load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if output == 'angry':
                self.consecutive_angry_count += 1
                if self.consecutive_angry_count >= 50:
                    # Video ekranına yazıyı ekle
                    cv2.putText(img, "Duygu durumunuz 'Sinirli' tespit edildi! Lutfen biraz sakinlesin.", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                self.consecutive_angry_count = 0

        return img


class DrowsinessAnalysis(VideoTransformerBase):
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.new_model = keras.models.load_model("drowsiness_model.h5")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)

        for x, y, w, h in eyes:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyess = self.eye_cascade.detectMultiScale(roi_gray)

            if len(eyess) == 0:
                print("Eyes are not detected")
            else:
                for (ex, ey, ew, eh) in eyess:
                    eyes_roi = roi_color[ey: ey + eh, ex:ex + ew]

        if 'eyes_roi' not in locals():
            eyes_roi = np.zeros((224, 224, 3), dtype=np.uint8)

        final_image = cv2.resize(eyes_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0

        Predictions = self.new_model.predict(final_image)

        if Predictions > 0.5:
            status = "Open Eyes"
            self.closed_eyes_count = 0  # Reset the count if eyes are open
        else:
            status = "Closed Eyes"
            self.closed_eyes_count += 1

            # Check if closed eyes count reaches 5
            if self.closed_eyes_count == 5:
                # Start a new thread to play the alert sound
                threading.Thread(target=play_alert_sound).start()

        faces = self.faceCascade.detectMultiScale(gray, 1.1, 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img,
            status,
            (50, 50),
            font,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_4
        )

        return img


class EmotionAndDrowsinessAnalysis(VideoTransformerBase):
    def __init__(self):
        # Emotion Detection
        self.emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}
        json_file = open('emotion_model1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.emotion_classifier = model_from_json(loaded_model_json)
        self.emotion_classifier.load_weights("emotion_model1.h5")

        # Drowsiness Detection
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.drowsiness_model = keras.models.load_model("drowsiness_model.h5")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Emotion Detection
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(img_gray, scaleFactor=1.3,
                                                                                              minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = self.emotion_classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = self.emotion_dict[maxindex]
                output = str(finalout)

            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if output == 'angry':
                self.consecutive_angry_count += 1
                if self.consecutive_angry_count >= 50:
                    # Video ekranına yazıyı ekle
                    cv2.putText(img, "Duygu durumu 'Sinirli' tespit edildi. Lutfen sakinlesin!", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                self.consecutive_angry_count = 0

        # Drowsiness Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)

        for x, y, w, h in eyes:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyess = self.eye_cascade.detectMultiScale(roi_gray)

            if len(eyess) == 0:
                print("Eyes are not detected")
            else:
                for (ex, ey, ew, eh) in eyess:
                    eyes_roi = roi_color[ey: ey + eh, ex:ex + ew]

        if 'eyes_roi' not in locals():
            eyes_roi = np.zeros((224, 224, 3), dtype=np.uint8)

        final_image = cv2.resize(eyes_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0

        Predictions = self.drowsiness_model.predict(final_image)

        if Predictions > 0.5:
            status = "Open Eyes"
            self.closed_eyes_count = 0  # Reset the count if eyes are open
        else:
            status = "Closed Eyes"
            self.closed_eyes_count += 1

            if self.closed_eyes_count == 5:
                threading.Thread(target=play_alert_sound).start()

        faces = self.faceCascade.detectMultiScale(gray, 1.1, 4)
        cv2.putText(
            img,
            status,
            (200, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_4
        )

        return img


def main():
    # Face Analysis Application #
    st.title("Real Time Driving Safety Application")
    activities = ["Drowsiness Detection", "Emotion Detection", "Emotion + Drowsiness Detection"]

    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(
        """
                 The application has three functionalities.

                 1. Real time Drowsiness detection.

                 2. Real time face emotion recognization.

                 3. Real time face emotion recognition and Drowsiness detection at the same time.

                 """
    )
    if choice == "Drowsiness Detection":
        st.write("Click on start to use webcam and detect your drowsiness")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_transformer_factory=DrowsinessAnalysis)
    elif choice == "Emotion Detection":
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)

    elif choice == "Emotion + Drowsiness Detection":
        st.write("Click on start to use webcam and detect your face emotion and drowsiness")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_transformer_factory=EmotionAndDrowsinessAnalysis)

    else:
        pass


if __name__ == "__main__":
    main()
