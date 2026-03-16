
import argparse
import cv2
import numpy as np
import tensorflow as tf
from preprocessing import preprocess_image

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion → color mapping (BGR)
EMOTION_COLORS = {
    'Angry':    (0,   0,   255),
    'Disgust':  (0,   128, 0  ),
    'Fear':     (128, 0,   128),
    'Happy':    (0,   255, 255),
    'Sad':      (255, 0,   0  ),
    'Surprise': (0,   165, 255),
    'Neutral':  (200, 200, 200),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/best_model.h5')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to static image. Omit for webcam mode.')
    parser.add_argument('--cascade', type=str,
                        default=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                        help='Haar Cascade XML path')
    return parser.parse_args()



# Face Detection (Viola-Jones / Haar Cascade)
def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces, gray



# Predict Emotion for a Single Face ROI
def predict_emotion(face_roi, model):
    processed = preprocess_image(face_roi, target_size=(48, 48))
    input_tensor = processed[np.newaxis, ..., np.newaxis]  # (1, 48, 48, 1)

    predictions = model.predict(input_tensor, verbose=0)[0]
    emotion_idx = np.argmax(predictions)
    confidence = predictions[emotion_idx]
    emotion = EMOTION_LABELS[emotion_idx]
    return emotion, confidence, predictions



# Draw Emotion Overlay on Frame
def draw_emotion(frame, x, y, w, h, emotion, confidence, predictions):
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Emotion label + confidence
    label = f"{emotion}: {confidence * 100:.1f}%"
    cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    # Mini bar chart for all emotions (top-right overlay)
    bar_x = frame.shape[1] - 200
    bar_y_start = 10
    for i, (emo, prob) in enumerate(zip(EMOTION_LABELS, predictions)):
        bar_y = bar_y_start + i * 25
        bar_len = int(prob * 180)
        bar_color = EMOTION_COLORS.get(emo, (200, 200, 200))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 18),
                      bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 180, bar_y + 18),
                      (100, 100, 100), 1)
        cv2.putText(frame, f"{emo[:3]}: {prob * 100:.0f}%",
                    (bar_x - 60, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    return frame



# Static Image Mode
def process_image(image_path, model, face_cascade):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    faces, gray = detect_faces(frame, face_cascade)

    if len(faces) == 0:
        print("[INFO] No faces detected.")
    else:
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            emotion, confidence, predictions = predict_emotion(face_roi, model)
            print(f"[RESULT] Detected: {emotion} ({confidence * 100:.1f}%)")
            frame = draw_emotion(frame, x, y, w, h, emotion, confidence, predictions)

    cv2.imshow('Facial Emotion Recognition', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Webcam Real-Time Mode
def process_webcam(model, face_cascade):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print("[INFO] Real-time detection started. Press 'q' to quit, 's' to save frame.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        faces, gray = detect_faces(frame, face_cascade)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            emotion, confidence, predictions = predict_emotion(face_roi, model)
            frame = draw_emotion(frame, x, y, w, h, emotion, confidence, predictions)

        # Status bar
        status = f"Frame: {frame_count} | Faces: {len(faces)} | Press Q=Quit S=Save"
        cv2.putText(frame, status, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Facial Emotion Recognition — Real Time', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f'saved_frame_{frame_count}.jpg'
            cv2.imwrite(fname, frame)
            print(f"[INFO] Frame saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection stopped.")



# Entry Point
def main():
    args = parse_args()

    print(f"[INFO] Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    face_cascade = cv2.CascadeClassifier(args.cascade)
    if face_cascade.empty():
        print(f"[ERROR] Failed to load Haar Cascade from: {args.cascade}")
        return

    if args.image:
        process_image(args.image, model, face_cascade)
    else:
        process_webcam(model, face_cascade)


if __name__ == "__main__":
    main()
