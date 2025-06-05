import pandas as pd

# Charger le fichier existant
df = pd.read_csv("dataset_final.csv")

# Séparer la colonne "label"
def extraire_labels(chaine):
    parties = chaine.split("_")
    
    # Vitesse = 1er élément
    vitesse = parties[0]

    # Mouvement = tout le reste
    mouvement = "_".join(parties[1:]) if len(parties) > 1 else "inconnu"
    
    return pd.Series([vitesse, mouvement])

# Appliquer la séparation
df[['vitesse', 'mouvement']] = df['label'].apply(extraire_labels)

# En option : réorganiser les colonnes
colonnes = df.columns.tolist()
colonnes.remove('label')
colonnes = [c for c in colonnes if c not in ['vitesse', 'mouvement']] + ['vitesse', 'mouvement']
df = df[colonnes]

# Sauvegarder
df.to_csv("dataset_final_split.csv", index=False)
print("Fichier mis à jour avec colonnes 'vitesse' et 'mouvement'")
