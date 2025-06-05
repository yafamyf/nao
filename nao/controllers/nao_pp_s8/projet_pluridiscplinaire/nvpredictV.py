# ───────────────────────── predictVitesse_MLP.py ─────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1) Lecture du jeu de données
df = pd.read_csv("vitesse_normalise_final.csv", sep=";")
print("Aperçu des données :")
print(df.head())

# 2) Encodage des classes (1 = lent, 2 = normal, 3 = rapide)
le = LabelEncoder()
df["vitesse_enc"] = le.fit_transform(df["vitesse"])
print("\nMapping :", dict(zip(le.classes_, le.transform(le.classes_))))

X = df.drop(columns=["vitesse", "vitesse_enc"])
y = df["vitesse_enc"]

# 3) Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nRépartition des classes (train) :")
print(pd.Series(y_train).value_counts(normalize=True))

# 4) Pipeline : StandardScaler + MLPClassifier
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp",    MLPClassifier(max_iter=500, random_state=42))
])

param_grid = {
    "mlp__hidden_layer_sizes": [(50,), (100,), (80, 40)],
    "mlp__alpha":             [1e-4, 1e-3, 1e-2],
    "mlp__learning_rate_init": [0.001, 0.01],
    "mlp__activation":        ["relu", "tanh"]
}

grid = GridSearchCV(
    pipe, param_grid, cv=5, n_jobs=-1, verbose=1, scoring="accuracy"
)
grid.fit(X_train, y_train)

print("\nMeilleurs hyperparamètres :", grid.best_params_)
best_model = grid.best_estimator_

# 5) Évaluation
y_pred = best_model.predict(X_test)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

# 6) Affichage matrice de confusion
labels = [f"{c} ({txt})" for c, txt in zip(le.transform(le.classes_), le.classes_)]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
ax.set_yticks(np.arange(len(labels)), labels=labels)
ax.set_xlabel("Prédictions")
ax.set_ylabel("Vraies classes")
ax.set_title("Matrice de confusion – MLP vitesse")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.tight_layout()
plt.show()

# 7) Sauvegarde du modèle
dump(best_model, "modele_vitesse_mlp.joblib")
print("\n✅ Modèle MLP sauvegardé dans 'modele_vitesse_mlp.joblib'")