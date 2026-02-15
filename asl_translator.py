import cv2
import hand_detector

video_capture = cv2.VideoCapture(0)

detector = hand_detector.HandDetector(
    max_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

while True:
    _, video_frame = video_capture.read()
    flip_frame = cv2.flip(video_frame, 1)

    landmarks_detected = False

    if not landmarks_detected:
        image, landmark, = detector.extract_landmarks(flip_frame)
        landmarks_detected = True
    else:
        landmarks_detected = False


    cv2.imshow('ASL Translator', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

