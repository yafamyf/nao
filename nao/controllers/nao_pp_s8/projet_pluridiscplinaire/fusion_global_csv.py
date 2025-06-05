import pandas as pd
import glob
import os

# Cherche tous les fichiers _combined.csv dans les sous-dossiers
fichiers = glob.glob("**/*_combined.csv", recursive=True)

frames = []

for fichier in fichiers:
    try:
        df = pd.read_csv(fichier)
        
        # Crée un label propre basé sur le chemin
        chemin = os.path.normpath(fichier)  # Normalise les séparateurs
        parties = chemin.split(os.sep)      # Sépare par dossier
        label_parts = [p for p in parties if "_combined" not in p and not p.endswith(".csv")]
        label = "_".join(label_parts)       # Assemble tous les dossiers menant au fichier

        df["label"] = label
        frames.append(df)
        print(f"✅ Chargé : {fichier} → label = {label}")
    except Exception as e:
        print(f"❌ Erreur avec {fichier} : {e}")

# Fusion
if frames:
    df_final = pd.concat(frames, ignore_index=True)
    df_final.to_csv("dataset_final.csv", index=False)
    print("🎯 Fusion réussie ! Taille :", df_final.shape)
else:
    print("⚠️ Aucun fichier trouvé.")
