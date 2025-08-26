import numpy as np
from Modèles.mlp import *
from Outils.initialisation import *
from Outils.visual_mlp import *

if __name__ == "__main__":
    # === 1. Chargement des données ===
    # Lecture du CSV "table_4_12.csv" où la dernière colonne contient les labels (0,1,...).
    X, y = load_classification_lastcol("../Datas/table_4_12.csv")
    print("X:", X.shape, "classes:", np.unique(y, return_counts=True))

    # === 2. Définition de l'architecture du réseau ===
    # Ici : nombre d'entrées = nombre de colonnes de X, deux couches cachées (12 et 10 neurones), 2 sorties.
    layers = [X.shape[1], 12, 10, 2]        # Structure du réseau
    activs = ["relu", "relu", "softmax"]    # Fonctions d'activation par couche

    # === 3. Entraînement du MLP ===
    W, b, losses, accs = fit_classification(
        X, y, layers, activs,
        lr=0.05, epochs=300, batch_size=None, reg_l2=1e-4, seed=1, verbose=True
    )

    # === 4. Courbe de perte (loss) ===
    plot_learning_curve(losses, "MLP – Courbe d'apprentissage (4.12)",
                        "../Plot/mlp/table_4_12_learning.png")

    # === 5. Courbe d'accuracy (précision sur l'entraînement) ===
    plot_accuracy_curve(accs, "MLP – Accuracy (4.12)",
                        "../Plot/mlp/table_4_12_accuracy.png")



    # === 6. Visualisation des régions de décision (2D) ===
    # On utilise un lambda pour passer la fonction de prédiction directement au traceur.
    # Les limites X et Y sont fixées à [-2, 2] avec un pas de tick de 0.5.
    plot_decision_regions_2d_fn(
        lambda Xg: predict_class(Xg, W, b, activs),
        X, y,
        title="MLP – Décision (4.12)",
        save_path="../Plot/mlp/table_4_12_decision.png",
        xlim=(-2, 2), ylim=(-2, 2), tick_step=0.5
    )



