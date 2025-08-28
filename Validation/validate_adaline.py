from Outils.initialisation import *
from Modèles.adaline import *
from Outils.visual_adaline import *

if __name__ == "__main__":
    # === 1. Chargement des données ===
    # Lecture du fichier CSV "table_2_11.csv"
    # - Pas d'en-tête (has_header=False)
    # - Colonne cible = dernière colonne (y_col=-1)
    x, y = load_data("../Datas/table_2_11.csv",has_header=False,y_col=-1)
    print("load_data: ok")

    # === 2. Ajout du biais (x0 = 1) ===
    # Ajout d'une colonne de 1 à gauche de X pour représenter le biais
    x_bias = add_x0(x)
    print("add_x0: ok")

    # === 3. Initialisation des poids ===
    # Poids initialisés à zéro (taille adaptée à x_bias)
    weight = ini_weight(x_bias)
    print("init_weight: ok")

    # === 4. Entraînement du modèle ADALINE ===
    # On entraîne le modèle avec :
    # - Taux d'apprentissage (lr) = 0.0001
    # - Nombre d'époques = 100
    # Retourne :
    #   - new_weight : poids appris
    #   - errors : liste des erreurs quadratiques moyennes (Emoy) par époque
    new_weight, errors = adaline_train(x_bias, y, lr=0.0001, epochs=100, mse_threshold=0.66)
    print("adaline_train: ok")

    # === 5. Prédictions du modèle ===
    # Prédictions sur l'ensemble d'entraînement avec les poids appris
    y_pred = adaline_predict(x_bias, new_weight)
    print("adaline_predict: ok")

    # === 6. Visualisation des résultats ===
    # Affiche :
    # - La droite de régression par rapport aux données
    # - La courbe d'apprentissage (erreur quadratique moyenne)
    plot_regression_result(x, y, y_pred, type="Adaline", title="Perceptron Adaline Table 2.11", save_plot=True)
    plot_learning_curve(errors, type="Adaline", title="Courbe d'aprentissage Adaline - 2.11", save_plot=True)


