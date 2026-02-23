import cv2
import mediapipe as mp
from sympy.codegen.ast import none


class HandDetector:

    def __init__(
            # Default Values
            self,
            max_hands: int = 1,
            min_detection_confidence: float = 0.8,
            min_tracking_confidence: float = 0.5,
    ):
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.handDetector = self.mpHands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.mpDrawing = mp.solutions.drawing_utils

    def extract_landmarks (self, image, padding=10):

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = self.handDetector.process(rgb_frame)
        landmarks_list = output.multi_hand_landmarks
        h, w, _ = image.shape

        hand_roi = None
        bbox = None

        if landmarks_list:
            for landmark in landmarks_list:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in landmark.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if y > y_max:
                        y_max = y
                    if x < x_min:
                        x_min = x
                    if y < y_min:
                        y_min = y
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Add padding and clamp to image boundaries
                x1 = max(0, x_min - padding)
                y1 = max(0, y_min - padding)
                x2 = min(w, x_max + padding)
                y2 = min(h, y_max + padding)

                # Extract ROI with padding
                hand_roi = image[y1:y2, x1:x2].copy()
                bbox = (x1, y1, x2, y2)

                self.mpDrawing.draw_landmarks(
                    image,
                    landmark,
                    self.mpHands.HAND_CONNECTIONS
                )

        return image, landmarks_list, hand_roi, bbox