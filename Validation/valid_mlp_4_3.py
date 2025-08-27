import numpy as np
from Modèles.mlp import fit_classification, predict_class
from Outils.initialisation import load_classification_lastcol
from Outils.visual_mlp import *
from Outils.visual_perceptron_3_5 import *

if __name__ == "__main__":
    # === 1. Chargement des données ===
    # Lecture du CSV "table_4_3.csv" où la dernière colonne contient le label.
    # La fonction gère aussi le passage de {-1,+1} à {0,1} si besoin.
    X, y = load_classification_lastcol("../Datas/table_4_3.csv")
    print("X:", X.shape, "classes:", np.unique(y, return_counts=True))

    # === 2. Définition de l’architecture du réseau ===
    # Le problème XOR est non-linéaire → nécessite au moins une couche cachée non-linéaire.
    # Ici : 2 entrées → 8 neurones cachés → 8 neurones cachés → 2 sorties.
    layers = [X.shape[1], 8, 8, 2]       # Taille des couches | x.shape donne la taille (les dimensions) du tableau x |
    activs = ["relu", "relu", "softmax"] # Fonctions d’activation par couche

    # === 3. Entraînement du réseau ===
    W, b, losses, accs = fit_classification(
        X, y, layers, activs,
        lr=0.05, epochs=300, batch_size=None, reg_l2=1e-4, seed=3, verbose=True
    )

    # === 4. Courbe de la fonction de coût (loss) ===
    plot_learning_curve(
        losses, "MLP – Courbe d'apprentissage (XOR 4.3)",
        "../Plot/mlp/table_4_3_learning.png"
    )

    # === 5. Courbe d'accuracy (précision sur l’entraînement) ===
    plot_accuracy_curve(
        accs, "MLP – Accuracy (XOR 4.3)",
        "../Plot/mlp/table_4_3_accuracy.png"
    )

    # === 6. Visualisation des régions de décision (2D) ===
    # On utilise un lambda pour passer une fonction de prédiction directement au plot.
    plot_decision_regions_2d_fn(
        lambda Xg: predict_class(Xg, W, b, activs), # Fonction de prédiction
        X, y,
        title="MLP – Décision (XOR 4.3)",
        save_path="../Plot/mlp/table_4_3_decision.png",
        xlim=(-0.5, 1.5), ylim=(-0.5, 1.5), tick_step=0.5
    )

    # === 7. Matrice de confusion normalisée ===
    # Évalue la capacité du réseau à bien classer chaque classe.
    y_pred = predict_class(X, W, b, activs)
    confusion_matrix_plot(
        y_true=y, y_pred=y_pred,
        class_names=["Classe 0", "Classe 1"],
        normalize="true",
        title="MLP – Matrice de confusion (XOR 4.3)",
        save_path="../Plot/mlp/table_4_3_confusion.png"
    )
