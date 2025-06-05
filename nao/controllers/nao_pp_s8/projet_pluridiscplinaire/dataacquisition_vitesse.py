import cv2
import joblib
import mediapipe as mp
import numpy as np
import datetime
from normFichiers import normaliser_bas_corps  # Assurez-vous que cette fonction est bien adaptée aux 20 colonnes

# Chargement du modèle depuis le fichier joblib
model_dict = joblib.load('modele_vitesse_rf_optimise.joblib')
model = model_dict['model'] if isinstance(model_dict, dict) else model_dict
print("✅ Modèle vitesse (Scikit-learn) chargé.")

min_detection_confidence = 0.5
min_tracking_confidence = 0.5
model_complexity = 1
video_source = 0
output_file = "test_vitesse.csv"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_source)

landmarks_c = (234, 63, 247)
connection_c = 240
thickness = 3
circle_r = 2
opened = True
tosave = []


landmark_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


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

def saveLandmarks(tosave, fn):
    with open(fn, "w") as f:
        for line in tosave:
            f.write(",".join(line) + "\n")
    print("💾 Landmarks sauvegardés dans", fn)

click_detected = False

def mouse_callback(event, x, y, flags, param):
    global click_detected
    if event == cv2.EVENT_RBUTTONDOWN:
        print("🖱️ Clic droit détecté.")
        click_detected = True

cv2.namedWindow("Media pipe pose detection")
cv2.setMouseCallback("Media pipe pose detection", mouse_callback)

def put_text_bottom(img, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2, color=(0, 0, 255)):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = img.shape[0] - 20
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

with mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                  min_tracking_confidence=min_tracking_confidence,
                  model_complexity=model_complexity) as pose:
    while opened:
        opened, image = cap.read()
        if not opened:
            break
        results = pose.process(image)
        output_img = image.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(output_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(landmarks_c, thickness, circle_r),
                                      mp_drawing.DrawingSpec(connection_c, thickness, circle_r))

        cv2.imshow("Media pipe pose detection", output_img)

        key = cv2.waitKey(10)

        if click_detected:
            click_detected = False
            if results.pose_landmarks:
                coords = []
                for idx in landmark_indices:
                    l = results.pose_landmarks.landmark[idx]
                    coords.append(l.x)
                    coords.append(l.y)

                keepLandmarks(results.pose_landmarks, tosave)

                X_input = np.array(coords).reshape(1, -1)
                X_scaled = normaliser_bas_corps(X_input)  # Normalisation bas du corps

                predicted_class = model.predict(X_scaled)[0]

                print(f"🏃 Vitesse prédite : {predicted_class}")

                image_with_text = output_img.copy()
                put_text_bottom(image_with_text, f"Vitesse : {predicted_class}")

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"vitesse_{timestamp}.png"
                cv2.imwrite(filename, image_with_text)
                print(f"📸 Image sauvegardée : {filename}")

        if key & 0xFF == ord("a"):
            if results.pose_landmarks:
                keepLandmarks(results.pose_landmarks, tosave)
        if key & 0xFF == ord("q") or key == 27:
            break

cap.release()
cv2.destroyAllWindows()
saveLandmarks(tosave, output_file)
