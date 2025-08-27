import numpy as np
from Outils.initialisation import add_x0, ini_weight

def adaline_train(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.0001,
    epochs: int = 200
) -> tuple[np.ndarray, list[float]]:
    """
    Entraîne un modèle ADALINE (Adaptive Linear Neuron) en utilisant la descente de gradient.

    Paramètres :
    ------------
    X : ndarray shape (n_samples, n_features)
        Les données d'entrée.
    y : ndarray shape (n_samples,)
        Les cibles réelles.
    lr : float
        Taux d'apprentissage (learning rate).
    epochs : int
        Nombre d'itérations sur l'ensemble du jeu de données.

    Retour :
    --------
    weights : ndarray shape (n_features+1,)
        Les poids appris (y compris le biais en première position).
    errors : list of float
        Liste des erreurs quadratiques moyennes (MSE/2) à chaque époque.
    """

    # Ajout du biais à gauche des données (x0 = 1)
    X_bias = add_x0(X)

    # Initialisation des poids à zéro (utilise la fonction ini_weight)
    weights = ini_weight(X_bias)

    # Liste pour stocker l'évolution de l'erreur
    errors = []

    # Boucle d'entraînement
    for epoch in range(epochs):

        # Calcul de la sortie du modèle (produit scalaire entre X et les poids)
        y_pred = X_bias.dot(weights)

        # Erreur entre la sortie attendue et la sortie prédite
        error = y - y_pred

        # Calcul de l'erreur quadratique moyenne divisée par 2
        mse = (error**2).mean() / 2
        errors.append(mse)

        # Calcul du gradient de l'erreur par rapport aux poids
        gradient = -X_bias.T.dot(error) / len(y)

        # Mise à jour des poids avec la descente de gradient
        weights -= lr * gradient

        print(f"Époque {epoch+1} | Emoy = {mse:.6f}")
    return weights, errors


def adaline_predict(
    X: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Prédit la sortie d'un modèle ADALINE entraîné.

    Paramètres :
    ------------
    X : ndarray shape (n_samples, n_features)
        Les données d'entrée.
    weights : ndarray shape (n_features+1,)
        Les poids du modèle (y compris le biais).

    Retour :
    --------
    y_pred : ndarray shape (n_samples,)
        Les valeurs prédites par le modèle.
    """

    # Ajout du biais à gauche des données
    X_bias = add_x0(X)

    # Produit scalaire entre X_bias et les poids
    return X_bias.dot(weights)
