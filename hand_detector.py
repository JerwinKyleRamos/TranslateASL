import cv2
import mediapipe as mp

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

    def extract_landmarks (self, image):

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = self.handDetector.process(rgb_frame)
        landmarks_list = output.multi_hand_landmarks

        if landmarks_list:
            for landmark in landmarks_list:
                self.mpDrawing.draw_landmarks(
                    image,
                    landmark,
                    self.mpHands.HAND_CONNECTIONS
                )

        return image, landmarks_list