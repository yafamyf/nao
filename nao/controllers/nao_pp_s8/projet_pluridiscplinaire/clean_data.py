import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv("dataset_final.csv")

# Supprimer les lignes incomplètes ou corrompues
df = df.dropna().drop_duplicates().reset_index(drop=True)

# Séparer les coordonnées et les labels
X_raw = df.drop("label", axis=1).copy()
y = df["label"].copy()

# Normalisation : recentrer sur les épaules et normaliser la taille
X_coords = X_raw.values.reshape(-1, 33, 2)  # 33 landmarks, 2 coordonnées

# Milieu épaules = (gauche + droite) / 2
center = (X_coords[:, 11, :] + X_coords[:, 12, :]) / 2
scale = np.linalg.norm(X_coords[:, 11, :] - X_coords[:, 12, :], axis=1).reshape(-1, 1, 1)

# Appliquer normalisation
X_normalized = (X_coords - center[:, np.newaxis, :]) / scale
X = X_normalized.reshape(X_normalized.shape[0], -1)  # Remettre en 2D

print("✅ Données prêtes :", X.shape)
