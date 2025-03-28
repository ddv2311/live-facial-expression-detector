import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('expression_detector.h5')

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Happy', 'Sad', 'Angry']

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print(f"Detected {len(faces)} faces")

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_resized = cv2.resize(roi_gray, (48, 48))
        roi_gray_normalized = roi_gray_resized.astype('float32') / 255.0
        roi_gray_input = np.expand_dims(roi_gray_normalized, axis=0)
        roi_gray_input = np.expand_dims(roi_gray_input, axis=-1)

        # Predict
        prediction = model.predict(roi_gray_input)
        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Debug output
        print(f"Raw prediction: {prediction[0]} (Happy, Sad, Angry)")
        print(f"Predicted: {emotion} ({confidence:.1f}%)")

        # Show processed face
        cv2.imshow('Processed Face', roi_gray_resized)

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Live Facial Expression Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

