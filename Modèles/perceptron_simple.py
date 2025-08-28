import numpy as np

def perceptron(
    X: np.ndarray,
    y: np.ndarray,
    X_bias: np.ndarray,
    weights: np.ndarray,
    learning_rate: float,
    max_epochs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Entraîne un perceptron simple (classification binaire).

    Paramètres :
    ------------
    X : ndarray shape (n_samples, n_features)
        Données d'entrée (sans biais ajouté).
    y : ndarray shape (n_samples,)
        Cibles réelles attendues (valeurs en {-1, 1}).
    X_bias : ndarray shape (n_samples, n_features+1)
        Données d'entrée avec la colonne de biais ajoutée en première position.
    weights : ndarray shape (n_features+1,)
        Poids initiaux du perceptron (y compris le biais).
    learning_rate : float
        Taux d'apprentissage (n).
    max_epochs : int
        Nombre maximum d'itérations complètes sur le jeu de données.

    Retour :
    --------
    X : ndarray
        Les données d'entrée initiales.
    y : ndarray
        Les cibles initiales.
    weights : ndarray
        Les poids appris après entraînement.
    """
    # Boucle principale d'entraînement sur plusieurs époques
    for epoch in range(max_epochs):
        nbErreurs = 0 # Compteur d'erreurs pour cette époque

        # Parcours de chaque échantillon du jeu d'entraînement
        for i in range(X_bias.shape[0]):

            # Calcul de la somme pondérée : w · x
            p = np.dot(X_bias[i], weights)

            # Application de la fonction d'activation (seuil) (y(k))
            y_pred = 1 if p >= 0 else -1

            # Calcul de l'erreur : valeur attendue - valeur prédite
            error = y[i] - y_pred #

            # Si la prédiction est incorrecte → mise à jour des poids
            if error != 0:
                # Règle de mise à jour du perceptron :
                # wi(t+1) = wi(t) + n * (d^k - y^k) * xi^k
                #   - wi(t+1) : nouveau poids
                #   - wi(t)   : ancien poids
                #   - n       : taux d'apprentissage (learning rate)
                #   - (d^k - y^k) : erreur (sortie attendue - sortie prédite)
                #   - xi^k    : valeur d'entrée pour la caractéristique i
                weights += learning_rate * error * X_bias[i] # pour la deuxième itération de l'époque 1  0.1*-2*[1,7,9]= nouveau poid -> [-2,-1,4,-1,8]
                nbErreurs += 1 # Incrémenter le compteur d'erreurs

    # Retourne les données et les poids appris
    return X, y, weights


