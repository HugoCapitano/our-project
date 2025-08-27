import numpy as np
import pandas as pd
from pathlib import Path
from Modèles.mlp import predict_class
from Outils.visual_mlp import plot_confusion_matrix

def class_to_letter(c: int) -> str:
    """
    Convertit un numéro de classe (1..5) en lettre (A..E).
    Exemple : 1 → A, 2 → B, ..., 5 → E
    """
    return "ABCDE"[int(c) - 1]  # c en 1..5

# === Définition des chemins principaux ===
root = Path("..") # Racine du projet
data_dir = root / "Datas" / "LangageDesSignes" # Dossier des datasets
val_path = data_dir / "validation_dataset.csv" # Dataset de validation

weights_path = root / "Outils" / "mlp_signs_weights.npz" # Poids du modèle sauvegardés après entraînement
plots_dir = root / "plot" / "mlp" / "sign" / "valid"     # Dossier pour les graphiques de validation
plots_dir.mkdir(parents=True, exist_ok=True)             # Création du dossier si nécessaire

# === Chargement des poids du modèle entraîné ===
d = np.load(weights_path)

"""# --- Uniquement a titre informatif ---
print(f"Les données enregistrée du modele : {d.files}")
print(f"Première couche : {d['W0'].shape}")
print(f"Première couche biais: {d['B0'].shape}")"""

# Reconstruction des listes de poids et biais
W = [d[f"W{k}"] for k in range(3)] # 3 couches dans ce MLP
B = [d[f"B{k}"] for k in range(3)]
acts = ["relu", "relu", "softmax"] # Fonctions d’activation par couche

# === Chargement du dataset de validation ===
val = pd.read_csv(val_path, header=None).to_numpy(dtype=float)
y_true_15 = val[:, 0].astype(int)        # Labels réels en 1..5 (A=1)
X_val     = val[:, 1:].astype(float)     # 42 features

# === Prédictions du modèle ===
y_pred_04 = predict_class(X_val, W, B, acts)  # 0..4
y_pred_15 = y_pred_04 + 1                     # 1..5

# === Calcul de l'accuracy ===
acc = (y_pred_15 == y_true_15).mean()
print(f"Accuracy: {acc:.3f}")

# === Matrice de confusion (affichage + sauvegarde) ===
plot_confusion_matrix(
    y_true=y_true_15,   # Labels réels
    y_pred=y_pred_15,   # Labels prédits
    labels=[1,2,3,4,5], # Liste des classes
    normalize=False,    # Comptes bruts (pas de normalisation)
    title="MLP – Matrice de confusion (validation)",
    save_path=str(plots_dir / "confusion_matrix.png")
)

# === Affichage détaillé des prédictions (optionnel) ===
print("\n--- Détails des prédictions (validation) ---")
for i, (yt, yp) in enumerate(zip(y_true_15, y_pred_15), start=1):
    features_str = ", ".join(f"{v:.4f}" for v in X_val[i-1])
    print(f"Ligne {i:02d}:")
    print(f"  Entrée (42 features): {features_str}")
    print(f"  Classe réelle : {yt} ({class_to_letter(yt)})")
    print(f"  Prédiction    : {yp} ({class_to_letter(yp)})\n")
