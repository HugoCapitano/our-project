from Outils.initialisation import load_data, add_x0, ini_weight
from Modèles.perceptron_simple import perceptron
from Outils.visual_perceptron_2D import plot_decision_boundary

if __name__ == "__main__":
    # === 1. Chargement des données ===
    # Lecture du fichier CSV "table_2_9.csv"
    # - Pas d'en-tête (has_header=False)
    # - Colonne cible = dernière colonne (y_col=-1)
    X, y = load_data("../Datas/table_2_9.csv", has_header=False, y_col=-1)
    print("load_data: ok")

    # === 2. Ajout du biais (x0 = 1) ===
    # On ajoute une colonne de 1 à gauche de X pour représenter le biais dans le modèle
    X_bias = add_x0(X)
    print("add_x0: ok")

    # === 3. Initialisation des poids ===
    # Les poids sont initialisés à 0 (taille adaptée à X_bias)
    w_init = ini_weight(X_bias)
    print("init_weight: ok")

    # === 4. Entraînement du perceptron ===
    # On entraîne le perceptron sur les données avec :
    # - learning_rate = 0.1
    # - max_epochs = 100
    new_X, new_y, new_w = perceptron(X, y, X_bias, w_init, learning_rate=0.1, max_epochs=100)
    print("perceptron_learning: ok")
    print(new_X, new_y, new_w)

    # === 5. Tracé de la frontière de décision ===
    # On affiche la séparation apprise par le perceptron
    # - X_has_bias=False car X ne contient pas la colonne x0
    # - bias_pos="first" car le biais est stocké au début du vecteur de poids
    # - type="perceptron_simple" pour organiser les sauvegardes
    # - save_plot=True pour sauvegarder l'image
    plot_decision_boundary(
        X, y, new_w,
        X_has_bias=False,
        bias_pos="first",  # mettre "last" si le biais est stocké en dernière position
        type="perceptron_simple",
        title="Simple Perceptron on table_2_9",
        save_plot=True
    )
