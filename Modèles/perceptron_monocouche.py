import numpy as np

# Flow entraînement (perceptron multi-classes)
# y -> one_hot  -> target
# xi -> scores = W @ xi -> predicted = argmax(scores)
# diff = target - predicted
# W += lr * diff[:, None] * xi
# accumulate error; repeat over samples and epochs; early stop if error=0


def one_hot_encode(y: np.ndarray, num_classes: int = None):
    """
    Convertit un vecteur de classes en représentation one-hot.

    Paramètres :
    ------------
    y : ndarray shape (n_samples,)
        Cibles réelles (valeurs entières représentant les classes).
    num_classes : int ou None
        Nombre total de classes. Si None, il est calculé automatiquement
        comme max(y) + 1.

    Retour :
    --------
    encoded : ndarray shape (n_samples, num_classes)
        Matrice one-hot : chaque ligne représente un échantillon avec un 1
        à la position de la classe, et 0 ailleurs.
    """
    if num_classes is None:
        num_classes = int(np.max(y)) + 1
        # Si num_classes est None, on prend la plus grande étiquette de y + 1 (on suppose que les classes sont des entiers >= 0).

    # Initialisation de la matrice de sortie
    encoded = np.zeros((y.size, num_classes))

    # Remplissage de la matrice : 1 à la position correspondant à la classe
    for idx, val in enumerate(y):
        encoded[idx, int(val)] = 1
    return encoded

    # Transforme les étiquettes (ex: 0,1,2) en vecteurs avec des 0 et 1 seul 1. Afin de bien gérer et identifier les classes.
    # Classe 0 → [1, 0, 0] / classe 1 → [0, 1, 0] / classe 2 → [0, 0, 1]
   

def perceptron_mono_train(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_classes: int,
    lr: float = 0.01,
    epochs: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Entraîne un perceptron monocouche pour un problème multi-classes
    avec encodage one-hot des cibles.

    Paramètres :
    ------------
    X : ndarray shape (n_samples, n_features)
        Données d'entrée.
    y : ndarray shape (n_samples,) 
        Cibles réelles (entiers représentant les classes).
    weights : ndarray shape (num_classes, n_features)
        Poids initiaux du modèle.
    num_classes : int
        Nombre total de classes.
    lr : float
        Taux d'apprentissage.
    epochs : int
        Nombre maximal d'époques.

    Retour :
    --------
    X : ndarray
        Données d'entrée initiales.
    y : ndarray
        Cibles initiales.
    weights : ndarray
        Poids appris après entraînement.
    errors : list of float
        Liste de l'erreur moyenne par époque.
    """
    # n_samples = nombre de données qu'on donnes au perceptron pour apprendre.

    # Encodage des cibles en one-hot
    y_encoded = one_hot_encode(y, num_classes)

    # Liste pour stocker l'évolution de l'erreur moyenne par époque
    errors = []

    # Boucle d'entraînement sur plusieurs époques
    for epoch in range(epochs):
        error_epoch = 0 # erreur cumulée pour l'époque

        # Parcours de chaque échantillon et de sa cible encodée
        for xi, target in zip(X, y_encoded):

            # Calcul des sorties pour chaque classe
            output = np.dot(weights, xi)

            # Prédiction binaire : 1 pour la classe ayant la sortie max, sinon 0 (en cas d'égalité, plusieurs 1 possibles - multi-hot)
            # Score = W · x → on prend la classe avec le score le plus élevé comme prédiction. Ex : [0.1, 0.7, 0.2] → classe 1 (car 0.7 max) 
            # Plus le score est grand, plus c’est probable que ce soit cette classe
            predicted = np.where(output == np.max(output), 1, 0)

            # Calcul de la mise à jour selon la règle du perceptron multi-classe
            # Si la prédiction est correcte, (target - predicted) sera 0, donc pas de mise à jour. Sinon, on ajuste les poids.
            update = lr * (target - predicted)

            # Mise à jour des poids pour toutes les classes
            # W ←W + η(t−t^)x⊤
            weights += update[:, np.newaxis] * xi

            # Accumulation de l'erreur quadratique (juste pour monitorer l'évolution de l'erreur)
            error_epoch += np.sum((target - predicted) ** 2)

        # Moyenne de l'erreur sur l'époque
        errors.append(error_epoch / len(y))

        # Arrêt anticipé si aucune erreur
        if error_epoch == 0:
            break

    # Retourne les données, les cibles, les poids appris et l'historique des erreurs
    return X, y, weights, errors


"""
xi = [2, 1]     # ses coordonnées
y  = 2          # vraie classe = 2 (jaune)
target = [0, 0, 1]   # one-hot de la classe 2

scores = [2, 1, -3]
predicted = [1, 0, 0]  # argmax de scores

comparaison :
target    = [0, 0, 1]
predicted = [1, 0, 0]
diff      = target - predicted
          = [-1, 0, +1]

"""