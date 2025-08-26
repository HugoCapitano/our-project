import numpy as np

def perceptron_gradient(X, y, X_bias, weights, learning_rate=0.01, max_epochs=100, emoy_threshold=0.01):
    """
    Implémente le perceptron avec la descente de gradient.
    Ici, l’activation est linéaire (identité), donc pas de fonction seuil.

    Paramètres :
    ------------
    X : ndarray shape (n_samples, n_features)
        Données d'entrée sans biais (non utilisé dans le calcul, juste pour compatibilité).
    y : ndarray shape (n_samples,)
        Cibles attendues (valeurs en {-1, 1}).
    X_bias : ndarray shape (n_samples, n_features+1)
        Données d'entrée avec biais ajouté en première colonne (x0 = 1).
    weights : ndarray shape (n_features+1,)
        Poids initiaux du modèle (y compris le biais).
    learning_rate : float
        Taux d'apprentissage (n).
    max_epochs : int
        Nombre maximum d'itérations sur l'ensemble du jeu de données.
    emoy_threshold : float
        Seuil de l’erreur quadratique moyenne pour arrêter l’apprentissage (convergence).

    Retour :
    --------
    X : ndarray
        Données d'entrée initiales (sans biais).
    y : ndarray
        Cibles initiales.
    weights : ndarray
        Poids appris après entraînement.
    emoy_list : list of float
        Liste des valeurs d'erreur quadratique moyenne (Emoy) à chaque époque.
    """

    # Nombre d'échantillons
    n_samples = X_bias.shape[0]

    # Liste pour stocker l’évolution de l’erreur quadratique moyenne
    emoy_list = []

    # Boucle d'entraînement sur plusieurs époques
    for epoch in range(max_epochs):

        # Accumulateur d'erreur totale pour cette époque
        total_error = 0.0

        # Parcours de chaque échantillon
        for i in range(n_samples):

            # Calcul de la sortie linéaire (pas de fonction seuil ici)
            y_pred = np.dot(X_bias[i], weights)

            # Erreur entre la sortie attendue et la sortie prédite
            error = y[i] - y_pred

            # Accumulation de l’erreur quadratique
            total_error += error ** 2

            # Mise à jour des poids selon la règle :
            # wi(t+1) = wi(t) + n * (d^k - y^k) * xi^k
            weights += learning_rate * error * X_bias[i]

        # Calcul de l'erreur quadratique moyenne Emoy
        emoy = total_error / (2 * n_samples)
        emoy_list.append(emoy)


        # Affichage si convergence atteinte (Emoy < seuil)
        if emoy < emoy_threshold:
            print(f"Convergence atteinte à l'époque {epoch + 1} avec Emoy = {emoy:.5f}")
            break

    # Retourne les données, les poids appris et la liste des erreurs moyennes
    return X, y, weights, emoy_list
