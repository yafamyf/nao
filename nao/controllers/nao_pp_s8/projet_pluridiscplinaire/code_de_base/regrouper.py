import pandas as pd
import os

# Chemin absolu corrigé
dossier = "."

# Récupère tous les fichiers dans le dossier
fichiers = [f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))]

frames = []

for fichier in fichiers:
    chemin_complet = os.path.join(dossier, fichier)
    try:
        df = pd.read_csv(chemin_complet)
        df["label"] = "lent_avancer"
        frames.append(df)
        print(f"Chargé : {fichier}")
    except Exception as e:
        print(f"Erreur avec {fichier}: {e}")

# Combine tous les fichiers
if frames:
    df_final = pd.concat(frames, ignore_index=True)
    df_final.to_csv("lent_avancer_combined.csv", index=False)
    print("Fusion terminée :", df_final.shape)
else:
    print("Aucun fichier valide trouvé.")
