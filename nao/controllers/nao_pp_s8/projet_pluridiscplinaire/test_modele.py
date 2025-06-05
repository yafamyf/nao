import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Chargement du CSV avec le bon séparateur
df = pd.read_csv("action.csv", sep=';')

# Indices à garder (0 à 24 landmarks, donc colonnes 0 à 49 inclus)
cols_to_keep = list(range(50))  # 25 points × 2 (x,y)

# Sélection uniquement des colonnes du haut du corps + bassin
X = df.iloc[:, cols_to_keep].values

# Séparation y (dernière colonne)
y = df.iloc[:, -1].values

print("Shape X:", X.shape)  # doit être (n_samples, 50)
print("Shape y:", y.shape)

def normaliser_par_epaules(X):
    """
    Centre les coordonnées x,y sur l'épaule gauche (colonne 22 pour x, 23 pour y)
    X est numpy array shape (n_samples, n_features)
    """
    X_norm = X.copy()
    x_coords = X[:, 0::2]
    y_coords = X[:, 1::2]

    # LEFT_SHOULDER_X est colonne 22 donc index 11 dans x_coords (22//2 = 11)
    ref_x = x_coords[:, 11]
    ref_y = y_coords[:, 11]

    x_coords = x_coords - ref_x[:, np.newaxis]
    y_coords = y_coords - ref_y[:, np.newaxis]

    X_norm[:, 0::2] = x_coords
    X_norm[:, 1::2] = y_coords
    return X_norm

# Normalisation
X_norm = normaliser_par_epaules(X)

# Encoder y s'il est catégoriel (string)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_enc, test_size=0.2, random_state=42)

# Création et entraînement du modèle MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

print(f"Score train: {mlp.score(X_train, y_train):.3f}")
print(f"Score test: {mlp.score(X_test, y_test):.3f}")

# Sauvegarde du modèle + label encoder dans un dict pour chargement futur
joblib.dump({'model': mlp, 'label_encoder': le}, 'modele_action.joblib')
print("✅ Modèle sauvegardé dans 'modele_action.joblib'")
