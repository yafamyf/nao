#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
)
from joblib import dump

df_action = pd.read_csv("action_normalise_final.csv", sep=";")

X = df_action.drop(columns=["geste"])
y = df_action["geste"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=200,          # un peu plus d’arbres pour la stabilité
    random_state=42,
    n_jobs=-1                  # parallélise l’entraînement
)
clf.fit(X_train, y_train)

probas  = clf.predict_proba(X_test)
classes = clf.classes_

best_thr, best_f1 = 0.0, -1.0
best_prec, best_rec = 0.0, 0.0

for thr in np.arange(0.50, 0.96, 0.02):          # 0.50 → 0.94
    preds = [
        classes[np.argmax(p)] if p.max() >= thr else "unknown"
        for p in probas
    ]
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, labels=["unknown"], zero_division=0
    )
    if f1[0] > best_f1:
        best_thr, best_f1 = thr, f1[0]
        best_prec, best_rec = prec[0], rec[0]

print(
    f"Seuil retenu : {best_thr:.2f}  |  "
    f"precision(unknown) = {best_prec:.2f}  |  "
    f"recall(unknown) = {best_rec:.2f}"
)

final_preds = [
    classes[np.argmax(p)] if p.max() >= best_thr else "unknown"
    for p in probas
]

print("\nRapport de classification - GESTES")
print(classification_report(y_test, final_preds, zero_division=0))

dump({"model": clf, "threshold": best_thr}, "modele_action_rf.joblib")
print("\n Modèle enregistré dans 'modele_action_rf.joblib' (avec seuil)")
