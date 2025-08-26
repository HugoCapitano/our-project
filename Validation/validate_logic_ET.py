from Outils.initialisation import load_data, add_x0, ini_weight
from Modèles.perceptron_simple import perceptron
from Outils.visual_perceptron_2D import plot_decision_boundary

if __name__ == "__main__":
    # === 1. Chargement des données ===
    # Lecture du fichier CSV "logic_et.csv"
    # - Pas d'en-tête (has_header=False)
    # - Colonne cible = dernière colonne (y_col=-1)
    X, y = load_data("../Datas/logic_et.csv", has_header=False, y_col=-1)
    print("load_data: ok")

    # === 2. Ajout du biais (x0 = 1) ===
    # Ajoute une colonne de 1 à gauche de X pour gérer le biais dans le calcul des poids
    X_bias = add_x0(X)
    print("add_x0: ok")

    # === 3. Initialisation des poids ===
    # Poids initialisés à zéro, taille adaptée à X_bias
    w_init = ini_weight(X_bias)
    print("init_weight: ok")

    # === 4. Entraînement du perceptron simple ===
    # Paramètres :
    # - learning_rate = 0.1
    # - max_epochs = 100
    # Retourne :
    #   - new_X, new_y : données et labels
    #   - new_w : poids finaux appris
    new_X, new_y, new_w = perceptron(X, y, X_bias, w_init, learning_rate=0.1, max_epochs=100)
    print("perceptron_learning: ok")
    print(new_X, new_y, new_w)

    # === 5. Visualisation ===
    # Tracer la frontière de décision sur les données originales (sans biais)
    plot_decision_boundary(
        X, y, new_w,
        X_has_bias=False,
        bias_pos="first",
        type="perceptron_simple",
        title="Simple Perceptron on logic_et",
        save_plot=True,
    )
