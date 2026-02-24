import csv
import os
import cv2
import hand_detector

cap = cv2.VideoCapture(0)

# Setup
dataset_path = 'dataset/landmark_dataset.csv'
SAMPLE_PER_CLASS = 400
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'del' , 'space']

# Initialize Hand Detector
detector = hand_detector.HandDetector(
    max_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

orientations = [
    'palm facing camera',
    'back of hand facing the camera'
]

# Create CSV with headers if it doesn't exist
if not os.path.exists(dataset_path):
    os.makedirs('dataset', exist_ok=True)
    with open(dataset_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(headers)

for class_name in CLASS_NAMES:
    print(f"\nGet ready to sign: {class_name}")
    print("Press SPACE to start collecting")

    for orientation in orientations:

        # Wait for spacebar or skip
        key = None
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            cv2.putText(frame, f"Current orientation: {orientation}", (12, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(frame, f"Next: {class_name}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('s'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        if key == ord('s'):
            print(f"Skipping {class_name}")
            continue

        # Collect samples
        count = 0
        print(f"Collecting {SAMPLE_PER_CLASS} samples for '{class_name}'...")

        while count < SAMPLE_PER_CLASS:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            image, landmark_list, hand_roi, bbox = detector.extract_landmarks(frame)

            if landmark_list:
                coords = []
                for lm in landmark_list[0].landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                with open(dataset_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([class_name] + coords)

                count += 1

            # Display progress on image (has landmarks drawn on it)
            cv2.putText(image, f"{class_name} - {orientation}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"{count}/{SAMPLE_PER_CLASS}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            progress = int((count / SAMPLE_PER_CLASS) * 400)
            cv2.rectangle(image, (10, 100), (410, 130), (50, 50, 50), -1)
            cv2.rectangle(image, (10, 100), (10 + progress, 130), (0, 255, 0), -1)

            cv2.imshow('Data Collection', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                exit()

        print(f" Done collecting {class_name}")

cap.release()
cv2.destroyAllWindows()
