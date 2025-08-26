import numpy as np
from Modèles.mlp import *
from Outils.initialisation import *
from Outils.visual_mlp import *

if __name__ == "__main__":
    # === 1. Chargement des données ===
    # Lecture du CSV "table_4_14.csv" où les 3 dernières colonnes sont les cibles en {-1, +1} (schéma one-vs-all).
    # La fonction renvoie X (features) et y (étiquettes 0..2).
    X, y = load_classification_n_classes("../Datas/table_4_14.csv", n_classes=3)
    print("X:", X.shape, "classes:", np.unique(y, return_counts=True))


    # === 2. Définition de l'architecture du réseau ===
    # - Nombre d'entrées = nombre de colonnes de X
    # - Deux couches cachées : 12 et 10 neurones
    # - 3 neurones en sortie (pour 3 classes)
    layers = [X.shape[1], 12, 10, 3]       # 3 neurones en sortie (3 classes)
    activs = ["relu", "relu", "softmax"]   # Softmax pour la classification multiclasse

    # === 3. Entraînement du MLP ===
    W, b, losses, accs = fit_classification(
        X, y, layers, activs,
        lr=0.03, epochs=1000, batch_size=None, reg_l2=1e-4, seed=1, verbose=True
    )

    # === 4. Tracé de la courbe de perte (loss) ===
    plot_learning_curve(losses, "MLP – Courbe d'apprentissage (4.14)",
                        "../Plot/mlp/table_4_14_learning.png")

    # === 5. Tracé de la courbe d'accuracy ===
    plot_accuracy_curve(accs, "MLP – Accuracy (4.14)",
                        "../Plot/mlp/table_4_14_accuracy.png")

    # === 6. Visualisation des zones de décision (2D) ===
    # - Utilisation d'une lambda pour injecter directement la prédiction dans la fonction de tracé
    # - Limites X et Y fixées à [-2, 2] avec ticks tous les 0.5
    plot_decision_regions_2d_fn(
        lambda Xg: predict_class(Xg, W, b, activs),
        X, y,
        title="MLP – Décision (4.14)",
        save_path="../Plot/mlp/table_4_14_decision.png",
        xlim=(-2, 2), ylim=(-2, 2), tick_step=0.5
    )
