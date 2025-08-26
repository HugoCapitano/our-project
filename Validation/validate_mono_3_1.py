import numpy as np
from Modèles.perceptron_monocouche import *
from Outils.visual_perceptron_mono import *
from Outils.initialisation import *

if __name__ == "__main__":
    # === 1. Chargement des données ===
    # Lecture du CSV "table_3_1.csv" (multi-classes)
    # - n_classes = 3 (problème à 3 classes)
    data="table_3_1.csv"
    X, y = load_classification_n_classes(f"../Datas/{data}",3)
    print("Données chargées. Classes:", np.unique(y))  # devrait afficher [0 1 2]

    # === 2. Ajout du biais (x0 = 1) ===
    X_bias = add_x0(X)
    print("add_x0 : ok")

    # === 3. Initialisation des poids ===
    # Initialisation à zéro d'une matrice de poids (num_classes lignes)
    num_classes = int(np.max(y) + 1)
    weights = ini_weight_multiclass(X_bias, num_classes=num_classes)
    print("Poids initialisés.")

    # === 4. Entraînement du perceptron monocouche ===
    # Paramètres :
    #   - X_bias : données avec biais
    #   - y : labels (0..num_classes-1)
    #   - weights : poids initiaux
    #   - lr : taux d'apprentissage
    #   - epochs : nombre maximal d'époques
    new_x, new_y, new_weights, errors = perceptron_mono_train(
        X_bias, y, weights, num_classes=num_classes, lr=0.1, epochs=100
    )
    print("Entraînement terminé.")
    print(f"Weights finaux:\n{new_weights}")

    # === 5. Visualisation ===
    # Affiche les régions de décision en 2D pour un problème multi-classes
    plot_decision_regions(
        X, y, new_weights,
        type="perceptron_monocouche",
        title=f"Perceptron_monocouche_region - {data} ",
        save=True
    )
