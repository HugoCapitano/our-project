import numpy as np

def one_hot_encode(y, num_classes=None):
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

    # Initialisation de la matrice de sortie
    encoded = np.zeros((y.size, num_classes))

    # Remplissage de la matrice : 1 à la position correspondant à la classe
    for idx, val in enumerate(y):
        encoded[idx, int(val)] = 1
    return encoded

def perceptron_mono_train(X, y, weights, num_classes, lr=0.01, epochs=100):
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

            # Prédiction binaire : 1 pour la classe ayant la sortie max, sinon 0
            predicted = np.where(output == np.max(output), 1, 0)

            # Calcul de la mise à jour selon la règle du perceptron multi-classe
            update = lr * (target - predicted)

            # Mise à jour des poids pour toutes les classes
            weights += update[:, np.newaxis] * xi

            # Accumulation de l'erreur quadratique
            error_epoch += np.sum((target - predicted) ** 2)

        # Moyenne de l'erreur sur l'époque
        errors.append(error_epoch / len(y))

        # Arrêt anticipé si aucune erreur
        if error_epoch == 0:
            break

    # Retourne les données, les cibles, les poids appris et l'historique des erreurs
    return X, y, weights, errors
