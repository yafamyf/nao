import cv2
import joblib
import mediapipe as mp
import numpy as np
import datetime

# Importer la fonction de normalisation qu'on a définie avant
def normaliser_par_epaules(X):
    X_norm = X.copy()
    x_coords = X[:, 0::2]
    y_coords = X[:, 1::2]

    # LEFT_SHOULDER_X = colonne 22 dans X donc index 11 dans x_coords
    ref_x = x_coords[:, 11]
    ref_y = y_coords[:, 11]

    x_coords = x_coords - ref_x[:, np.newaxis]
    y_coords = y_coords - ref_y[:, np.newaxis]

    X_norm[:, 0::2] = x_coords
    X_norm[:, 1::2] = y_coords
    return X_norm

# Chargement du modèle et label encoder
model_dict = joblib.load('modele_action.joblib')
model = model_dict['model']
le = model_dict['label_encoder']
print("✅ Modèle et encodeur chargés.")

landmark_indices = list(range(25))

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

landmarks_c = (234, 63, 247)
connection_c = 240
thickness = 3
circle_r = 2
opened = True
tosave = []


def keepLandmarks(landmarks, tosave):
    if len(tosave) == 0:
        header = []
        for i, l in enumerate(landmarks.landmark):
            header.append(f"{mp_pose.PoseLandmark(i).name}_X")
            header.append(f"{mp_pose.PoseLandmark(i).name}_Y")
        tosave.append(header)
    line = []
    for l in landmarks.landmark:
        line.append(str(l.x))
        line.append(str(l.y))
    tosave.append(line)
    print("✅ Coordonnées ajoutées à la liste à sauvegarder.")

def put_text_bottom(img, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2, color=(0, 0, 255)):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = img.shape[0] - 20
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

click_detected = False
def mouse_callback(event, x, y, flags, param):
    global click_detected
    if event == cv2.EVENT_RBUTTONDOWN:
        print("🖱️ Clic droit détecté.")
        click_detected = True

cv2.namedWindow("MediaPipe Pose Detection")
cv2.setMouseCallback("MediaPipe Pose Detection", mouse_callback)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5,
                  model_complexity=1) as pose:

    while opened:
        opened, image = cap.read()
        if not opened:
            break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        output_img = image.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(output_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(landmarks_c, thickness, circle_r),
                                      mp_drawing.DrawingSpec(connection_c, thickness, circle_r))

        cv2.imshow("MediaPipe Pose Detection", output_img)

        key = cv2.waitKey(10)

        if click_detected:
            click_detected = False
            if results.pose_landmarks:
                coords = []
                for idx in landmark_indices:
                    lm = results.pose_landmarks.landmark[idx]
                    coords.append(lm.x)
                    coords.append(lm.y)

                X_input = np.array(coords).reshape(1, -1)
                X_scaled = normaliser_par_epaules(X_input)

                pred_enc = model.predict(X_scaled)[0]
                predicted_class = le.inverse_transform([pred_enc])[0]

                print(f"🎯 Classe prédite : {predicted_class}")

                img_with_text = output_img.copy()
                put_text_bottom(img_with_text, f"Classe : {predicted_class}")

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"prediction_{timestamp}.png"
                cv2.imwrite(filename, img_with_text)
                print(f"📸 Image sauvegardée : {filename}")

        if key & 0xFF == ord("q") or key == 27:
            break

cap.release()
cv2.destroyAllWindows()
