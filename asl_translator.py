from collections import deque
import cv2
import numpy as np
import hand_detector
import tensorflow as tf
import json
import time

video_capture = cv2.VideoCapture(0)

# Load Model
try:
    model = tf.keras.models.load_model('model_outputs/best_model.keras')
    print(f"Model loaded {model}")

    with open('model_outputs/class_names.json', 'r') as f:
        class_names = json.load(f)
        print(f"Class names loaded {class_names}")

except FileNotFoundError as e:
    exit(1)

# Configuration
IMG_SIZE = 96
CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 3


# Prediction smoothing
class PredictionSmoother:
    def __init__(self, window_size=5):
        self.predictions = deque(maxlen=window_size)

    def add(self, prediction):
        self.predictions.append(prediction)

    def get(self):
        if len(self.predictions) == 0:
            return None
        return np.mean(self.predictions, axis=0)

    def reset(self):
        self.predictions.clear()


smoother = PredictionSmoother(SMOOTHING_WINDOW)


# Initialize Hand Detector
detector = hand_detector.HandDetector(
    max_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

# FPS calculation variables
ptime = 0 # previous time
ctime = 0 # current time

translation = ""
last_prediction = None
prediction_start_time = None
hold_time = 1.5

prev_wrist_x = None
prev_wrist_y = None

while True:
    _, video_frame = video_capture.read()
    flip_frame = cv2.flip(video_frame, 1)

    image, landmark_list, hand_roi, bbox = detector.extract_landmarks(flip_frame)

    # Calculate and Displays Frame FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if hand_roi is not None and bbox is not None :

        x1, y1, x2, y2 = bbox

        # Check if hand is moving
        wrist = landmark_list[0].landmark[0]
        is_moving = False

        if prev_wrist_x is not None:
            dx = abs(wrist.x - prev_wrist_x)
            dy = abs(wrist.y - prev_wrist_y)
            is_moving = (dx + dy) > 0.015

        prev_wrist_x = wrist.x
        prev_wrist_y = wrist.y

        if is_moving:
            smoother.reset()  # clear smoothing buffer when moving
            cv2.putText(image, "Hold still...", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Extract landmarks
            coords = []
            for lm in landmark_list[0].landmark:
                coords.extend([lm.x, lm.y, lm.z])
            input_data = np.array(coords).reshape(1, -1)
            prediction = model.predict(input_data, verbose=0)[0]
            smoother.add(prediction)
            smooth_pred = smoother.get()

            if smooth_pred is not None:
                idx = np.argmax(smooth_pred)
                conf = smooth_pred[idx]
                pred_class = class_names[idx]

                if conf >= CONFIDENCE_THRESHOLD:
                    if pred_class != "nothing" and pred_class != last_prediction:
                        prediction_start_time = time.time()
                        last_prediction = pred_class
                    elif pred_class == last_prediction and prediction_start_time:
                        elapsed = time.time() - prediction_start_time

                        if elapsed >= hold_time:
                            if pred_class == "del" and len(translation) > 0:
                                translation = translation[:-1]
                            elif pred_class == "space":
                                translation += " "
                            elif pred_class != "del":
                                translation += pred_class

                            prediction_start_time = None
                            last_prediction = None
                            print(f"Translation: {translation}")
                            # Bounding Box
                            cv2.putText(image, f"{pred_class} ({conf:.2f})", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            print(f"Pred: {pred_class} ({conf:.2f})")
                            print(f"Top 5: {[(class_names[i], round(smooth_pred[i], 2)) for i in np.argsort(smooth_pred)[-5:][::-1]]}")

    # Display translation
    h, w, _ = image.shape
    cv2.rectangle(image, (0, h - 70), (w, h), (0, 0, 0), -1)
    cv2.putText(image, f"Translation: {translation}", (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, "space: Reset | Q: Quit", (10, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('ASL Translator', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

