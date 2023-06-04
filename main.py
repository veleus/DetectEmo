import cv2
from fer import FER

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.detector = FER()

    def detect_emotions_in_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = frame[y:y + h, x:x + w]
            emotions = self.detector.detect_emotions(face)

            if len(emotions) > 0:
                emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

cap = cv2.VideoCapture(0)

emotion_detector = EmotionDetector()

while True:
    ret, frame = cap.read()

    frame_with_emotions = emotion_detector.detect_emotions_in_frame(frame)

    cv2.imshow('Emotion Detection', frame_with_emotions)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
