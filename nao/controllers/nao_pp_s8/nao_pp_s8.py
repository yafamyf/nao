# ──────────────────────────────────────────────────────────────
# Contrôleur Nao : reconnaissance du geste + estimation de la
# vitesse (MLP) et déclenchement des mouvements
# ──────────────────────────────────────────────────────────────

from controller import Robot
from naomotion import NaoMotion

import cv2
import numpy as np
import joblib

# ────────────────────────
# MediaPipe : fallback selon l’install
try:
    from mediapipe.solutions import pose as mp_pose
    from mediapipe.solutions import drawing_utils as mp_draw
except ModuleNotFoundError:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_draw

# ────────────────────────
# Chargement des modèles
geste_bundle   = joblib.load("projet_pluridiscplinaire/modele_action.joblib")
geste_model    = geste_bundle["model"]
label_encoder  = geste_bundle["label_encoder"]

vitesse_bundle = joblib.load("projet_pluridiscplinaire/modele_vitesse_mlp.joblib")
if isinstance(vitesse_bundle, dict):
    vitesse_model = vitesse_bundle["model"]
else:
    vitesse_model = vitesse_bundle

# ──────────────────────────────────────────────────────────────
# Mapping vitesse (classes → coefficients de vitesse)
VITESSE_MAPPING = {
    0: 0.3,  # lent
    1: 0.6,  # normal
    2: 1.0   # rapide
}

# Correspondance entre les libellés du jeu de données et
# ceux utilisés pour commander les mouvements du robot.
GESTE_MAPPING = {
    "pas de cote_gauche": "fairePasGauche",
    "pas de cote_droite": "fairePasDroite",
    "tourner_gauche": "tournerGauche",
    "tourner_droite": "tournerDroite",
}

def normaliser_geste(label: str) -> str:
    """Renvoie un nom de geste compréhensible par NaoMotion."""
    return GESTE_MAPPING.get(label, label)

# Normalisation par l’épaule gauche
def normaliser_par_epaules(X):
    X_norm  = X.copy()
    x_all   = X[:, 0::2]
    y_all   = X[:, 1::2]
    ref_x   = x_all[:, 11]  # idx 11 = LEFT_SHOULDER.x
    ref_y   = y_all[:, 11]

    x_all -= ref_x[:, None]
    y_all -= ref_y[:, None]

    X_norm[:, 0::2] = x_all
    X_norm[:, 1::2] = y_all
    return X_norm

def extract_landmarks(flat_vec, indices):
    """Extrait les coordonnées x,y des indices MediaPipe demandés"""
    flat_vec = flat_vec.reshape(-1)
    coords = []
    for idx in indices:
        coords.extend([flat_vec[2*idx], flat_vec[2*idx + 1]])
    return np.array(coords).reshape(1, -1)

# Landmarks bas du corps MediaPipe
LANDMARKS_BAS_CORPS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# ──────────────────────────────────────────────────────────────
# Setup robot + webcam
robot    = Robot()
TIMESTEP = int(robot.getBasicTimeStep())
motion   = NaoMotion(robot)

mp_pose_estimator = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# ──────────────────────────────────────────────────────────────
# Boucle principale
try:
    while robot.step(TIMESTEP) != -1:
        ok, frame = cap.read()
        if not ok:
            print("❌ Impossible de lire la webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = mp_pose_estimator.process(frame_rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

            # Extraction complète des 33 landmarks (x,y) → 66 features
            coords66 = []
            for i in range(33):
                lm = results.pose_landmarks.landmark[i]
                coords66.extend([lm.x, lm.y])
            vec66 = np.array(coords66).reshape(1, -1)
            vec66_norm = normaliser_par_epaules(vec66)

            # Sous-vecteur pour geste (les 25 premiers landmarks → 50 features)
            vec50_norm = vec66_norm[:, :50]

            # Sous-vecteur pour vitesse (bas du corps uniquement)
            vec16_norm = extract_landmarks(vec66_norm, LANDMARKS_BAS_CORPS)

            # Prédiction geste
            geste_enc = geste_model.predict(vec50_norm)[0]
            geste_label = label_encoder.inverse_transform([geste_enc])[0]
            geste_label = normaliser_geste(geste_label)

            # Prédiction vitesse
            vitesse_enc = vitesse_model.predict(vec16_norm)[0]
            vitesse_fac = VITESSE_MAPPING.get(int(vitesse_enc), 0.5)

            vitesse_int = int(vitesse_enc) + 1


            print(f"🎯 Geste : {geste_label:<15} | Vitesse (coeff) : {vitesse_fac:.2f} | Vitesse (int) : {vitesse_int}")

            # Commande robot
            if geste_label == "avancer":
                motion.forward(vitesse_int)
            elif geste_label == "reculer":
                motion.backward(vitesse_int)
            elif geste_label == "tournerGauche":
                motion.turn("left", vitesse_int)
            elif geste_label == "tournerDroite":
                motion.turn("right", vitesse_int)
            elif geste_label == "fairePasGauche":
                motion.sidestep("left", vitesse_int)
            elif geste_label == "fairePasDroite":
                motion.sidestep("right", vitesse_int)

        cv2.imshow("Webcam + Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print("❌ Erreur dans le contrôleur :", e)

finally:
    cap.release()
    cv2.destroyAllWindows()
