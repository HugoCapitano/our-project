from pathlib import Path
import numpy as np
import pandas as pd

# — imports projet —
from Modèles.mlp import fit_classification, predict_class
from Outils.visual_mlp import plot_learning_curve, plot_accuracy_curve

# ----- Définition des chemins -----
root = Path("..") # Racine du projet
data_dir = root / "Datas" / "LangageDesSignes"   # Dossier contenant les datasets
learn_path = data_dir / "learning_dataset.csv"   # Dataset d'apprentissage
val_path   = data_dir / "validation_dataset.csv" # Dataset de validation

# Dossier où seront sauvegardées les courbes d'entraînement
plots_dir = root / "plot" / "mlp" / "sign" / "train"
plots_dir.mkdir(parents=True, exist_ok=True)

# Fichier où sauvegarder les poids du modèle entraîné
weights_path = (root / "Outils" / "mlp_signs_weights.npz")
weights_path.parent.mkdir(parents=True, exist_ok=True)

# ----- Chargement du dataset d'apprentissage -----
# Format : première colonne = label (1..5), 42 colonnes suivantes = features
learn = pd.read_csv(learn_path, header=None).to_numpy(dtype=float)

# Conversion des labels de 1..5 vers 0..4 (utilisé par le modèle)
y_train = learn[:, 0].astype(int) - 1

# Extraction des features (42 colonnes)
X_train = learn[:, 1:].astype(float)

# ----- Configuration et entraînement du MLP -----
layers = [42, 64, 32, 5]  # Architecture : entrée, 2 couches cachées, sorti
acts   = ["relu", "relu", "softmax"] # Fonctions d'activation

W, B, losses, accs = fit_classification(
    X_train, y_train, layers, acts,
    lr=0.05,        # Taux d'apprentissage
    epochs=400,     # Nombre d'époques
    batch_size=32,  # Taille du mini-batch
    reg_l2=1e-4,    # Régularisation L2
    seed=7,         # Graine pour reproductibilité
    verbose=True    # Affichage des infos d'entraînement
)

# ----- Sauvegarde des poids appris -----
# Chaque couche est sauvegardée séparément sous forme W0, B0, W1, B1, etc.
np.savez(
    weights_path,
    **{f"W{k}": W[k] for k in range(len(W))},
    **{f"B{k}": B[k] for k in range(len(B))}
)
print(f"→ Poids sauvegardés dans {weights_path}")

# ----- Tracé et sauvegarde des courbes d'entraînement -----
# Courbe de la perte (loss) par époque
plot_learning_curve(
    losses,
    title="MLP – Courbe d'apprentissage (loss)",
    save_path=str(plots_dir / "learning_curve_loss.png")
)

# Courbe de l'accuracy par époque
plot_accuracy_curve(
    accs,
    title="MLP – Courbe d'accuracy (train)",
    save_path=str(plots_dir / "learning_curve_accuracy.png")
)
