import numpy as np

def perceptron_gradient(
    X: np.ndarray,
    y: np.ndarray,
    X_bias: np.ndarray,
    weights: np.ndarray,
    learning_rate: float = 0.01,
    max_epochs: int = 100,
    emoy_threshold: float = 0.01
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Implémente le perceptron avec la descente de gradient FULL BATCH.
    Ici, l’activation est linéaire (identité), donc pas de fonction seuil.
    """
    n_samples = X_bias.shape[0] # Nombre d'échantillons
    emoy_list = [] # Liste pour stocker l'erreur quadratique moyenne à chaque époque

    for epoch in range(max_epochs): # Boucle sur le nombre d'époques
        # Prédiction sur tout le batch (calcul linéaire)

        y_pred = X_bias.dot(weights) # Calcul des sorties du perceptron

        error = y - y_pred # Calcul de l'erreur entre la cible et la prédiction

        # Erreur quadratique moyenne (MSE/2)
        emoy = (error ** 2).mean() / 2 
        emoy_list.append(emoy) # Stocke l'erreur pour suivi

        # Calcul du gradient global (vectorisé sur tout le batch) 
        gradient = -X_bias.T.dot(error) / n_samples # Gradient de l'erreur 

        # Mise à jour des poids (une seule fois par époque)
        weights -= learning_rate * gradient

        # Condition d'arrêt si la perte est suffisamment faible

        if emoy < emoy_threshold:
            print(f"Convergence atteinte à l'époque {epoch + 1} avec Emoy = {emoy:.5f}")
            break

    # Retourne les données, les cibles, les poids appris et la liste des erreurs
    return X, y, weights, emoy_list