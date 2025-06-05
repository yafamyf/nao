import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from joblib import load

# Chargement du modèle
clf_vitesse = load("modele_vitesse_mlp.joblib")

# Seuil de confiance
seuil = 0.73
classes = clf_vitesse.classes_

# Colonnes MediaPipe du bas du corps
all_columns = [
    'LEFT_HIP_X', 'LEFT_HIP_Y',
    'RIGHT_HIP_X', 'RIGHT_HIP_Y',
    'LEFT_KNEE_X', 'LEFT_KNEE_Y',
    'RIGHT_KNEE_X', 'RIGHT_KNEE_Y',
    'LEFT_ANKLE_X', 'LEFT_ANKLE_Y',
    'RIGHT_ANKLE_X', 'RIGHT_ANKLE_Y',
    'LEFT_HEEL_X', 'LEFT_HEEL_Y',
    'RIGHT_HEEL_X', 'RIGHT_HEEL_Y',
    'LEFT_FOOT_INDEX_X', 'LEFT_FOOT_INDEX_Y',
    'RIGHT_FOOT_INDEX_X', 'RIGHT_FOOT_INDEX_Y'
]

def normaliser_bas_corps(row):
    keypoints = row.values.astype(np.float32).reshape(-1, 2)
    # keypoints[0] = LEFT_HIP (landmark 23)
    # keypoints[1] = RIGHT_HIP (landmark 24)
    # keypoints[2] = LEFT_KNEE (landmark 25)
    # keypoints[3] = RIGHT_KNEE (landmark 26)
    # etc.

    centre = (keypoints[0] + keypoints[1]) / 2  # milieu des hanches
    dist1 = np.linalg.norm(keypoints[0] - keypoints[2])  # LEFT_HIP -> LEFT_KNEE
    dist2 = np.linalg.norm(keypoints[1] - keypoints[3])  # RIGHT_HIP -> RIGHT_KNEE
    scale = (dist1 + dist2) / 2 if (dist1 + dist2) > 0 else 1e-5

    keypoints = (keypoints - centre) / scale
    return keypoints.flatten()


# Initialisation MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Impossible d'accéder à la caméra")
    exit()

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5,
                  model_complexity=1) as pose:

    print("Caméra ouverte, appuie sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Impossible de lire la frame")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        output = frame.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extraire uniquement les points du bas du corps
            keypoints = []
            for i in range(23, 33):
                lm = results.pose_landmarks.landmark[i]
                keypoints.append(lm.x)
                keypoints.append(lm.y)

            df = pd.DataFrame([keypoints], columns=all_columns)
            row = df.iloc[0]
            X_test = pd.DataFrame([normaliser_bas_corps(row)])

            probas = clf_vitesse.predict_proba(X_test)
            max_proba = np.max(probas)
            pred_class = np.argmax(probas)
            label = classes[pred_class] if max_proba >= seuil else "unknown"

            cv2.putText(output, f"Geste : {label} ({max_proba:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if label != "unknown" else (0, 0, 255), 2)

        cv2.imshow("Reconnaissance de gestes", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
