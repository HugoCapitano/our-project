from Modèles.perceptron_monocouche import *
from Outils.initialisation import *
from Outils.visual_perceptron_3_5 import *


if __name__ == "__main__":
    # === 1. Chargement des données ===
    # Lecture du CSV "table_3_5.csv"
    # - n_classes dernières colonnes encodées en {-1, +1} (schéma one-vs-all)
    data="table_3_5.csv"
    X, y = load_data_3_5("../Datas/table_3_5.csv", n_classes=4)
    print("Données 3_5 chargées. Classes:", np.unique(y))

    # === 2. Ajout du biais (x0 = 1) ===
    X_bias = add_x0(X)
    print("add_x0 : ok")

    # === 3. Initialisation des poids ===
    # Poids initialisés à zéro pour chaque neurone de sortie (1 ligne par classe)
    num_classes = 4
    weights = ini_weight_multiclass(X_bias, num_classes=num_classes)
    print("Poids initialisés (shape:", weights.shape, ")")

    # === 4. Entraînement du perceptron monocouche ===
    X_tr, y_tr, W, errors = perceptron_mono_train(
        X_bias, y, weights, num_classes=num_classes, lr=0.1, epochs=200
    )
    print("Entraînement terminé.")
    print("Weights finaux:\n", W)

    # === 5. Évaluation de la performance ===
    # Prédiction sur l’ensemble d’entraînement
    y_pred = predict(X_bias, W)
    acc = (y_pred == y).mean()
    print(f"Accuracy entraînement: {acc:.3f}")

    # Accuracy par classe
    for c in range(num_classes):
        cls_mask = (y == c)
        if cls_mask.any():
            acc_c = (y_pred[cls_mask] == y[cls_mask]).mean()
            print(f"  - Classe {c}: {acc_c:.3f} (n={cls_mask.sum()})")

    # === 6. Matrice de confusion ===
    # Affichage et sauvegarde de la matrice de confusion
    class_names = ["Symbole 0", "Symbole 1", "Symbole 2", "Symbole 3"]
    cm = confusion_matrix_plot(
        y_true=y,
        y_pred=y_pred,
        class_names=class_names,
        normalize=None,
        type="perceptron_monocouche",
        title=f"Perceptron_monocouche_maxtrix - {data} ",
        figsize=(6.5, 6.5),
        save = True
    )
    print("Matrice de confusion :\n", cm)

    # === 7. Visualisation des prototypes ===
    # Affiche un symbole "moyen" par classe (moyenne des pixels)
    plot_class_prototypes_csv(
        "../Datas/table_3_5.csv",
        n_classes=4,
        grid_shape=(5,5),
        save=True,
        type="perceptron_monocouche",
        title=f"Perceptron_monocouche_symbols - {data} "
    )
