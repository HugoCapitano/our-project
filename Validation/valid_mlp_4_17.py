from Modèles.mlp import *
from Outils.initialisation import *
from Outils.visual_mlp import *


# === Fonctions utilitaires pour la standardisation (z-score) ===
def zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule la moyenne et l'écart-type de chaque feature pour la standardisation z-score.
    Paramètres :
    ------------
    X : ndarray shape (m, n)
        Données d'entrée.
    Retour :
    --------
    mu : ndarray shape (1, n)
        Moyenne de chaque feature.
    sd : ndarray shape (1, n)
        Écart-type de chaque feature.
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sd

def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """
    Applique la standardisation z-score aux données X.
    Paramètres :
    ------------
    X : ndarray shape (m, n)
        Données d'entrée.
    mu : ndarray shape (1, n)
        Moyenne de chaque feature (issue de zscore_fit).
    sd : ndarray shape (1, n)
        Écart-type de chaque feature (issue de zscore_fit).
    Retour :
    --------
    X_std : ndarray shape (m, n)
        Données standardisées.
    """
    return (X - mu) / sd

if __name__ == "__main__":
    # === 1. Chargement des données ===
    # Lecture du CSV "table_4_17.csv" où la dernière colonne est la cible y.
    X_raw, y_raw = load_regression_mlp("../Datas/table_4_17.csv")
    y_raw = y_raw.reshape(-1, 1)


    # === 2. Standardisation des données ===
    # Important pour les MLP avec tanh : améliore la convergence.
    # Standardisation des entrées X
    x_mu, x_sd = zscore_fit(X_raw)
    X = zscore_apply(X_raw, x_mu, x_sd)

    # Standardisation de la sortie y (pour stabiliser l'apprentissage)
    y_mu, y_sd = y_raw.mean(0, keepdims=True), y_raw.std(0, keepdims=True) + 1e-8
    y = (y_raw - y_mu) / y_sd


    # === 3. Définition de l'architecture du MLP ===
    # - Deux couches cachées avec 32 neurones chacune
    # - Activation tanh pour les couches cachées
    # - Activation linéaire en sortie (régression)
    layers = [X.shape[1], 32, 32, 1]
    activs = ["tanh", "tanh", "linear"]

    # === 4. Entraînement du MLP ===
    W, b, losses = fit_regression(
        X, y, layers, activs,
        lr=0.004, epochs=5000, batch_size=32, reg_l2=1e-4, seed=3, verbose=True
    )

    # Tracé de la courbe de perte
    plot_learning_curve(losses, "MLP – Courbe d'apprentissage (4.17)",
                        "../Plot/mlp/table_4_17_learning.png")

    # === 5. Courbe de régression lissée ===
    # On encapsule la standardisation/déstandardisation pour la prédiction
    def predict_fn_xraw(Xg_raw):
        Xg = zscore_apply(Xg_raw, x_mu, x_sd)           # Standardisation des entrées
        yhat_std = predict_regression(Xg, W, b, activs) # Prédiction normalisée
        return yhat_std * y_sd + y_mu                   # Déstandardisation

    plot_regression_4_17(
        predict_fn_xraw, X_raw, y_raw,
        feature_index=0, n_points=900,
        xlim=(-2, 2), ylim=(-1.5, 1.8), tick_step=0.5,
        title="MLP Non-Linear Regression Fit (4.17)",
        save_path="../Plot/mlp/table_4_17_fit.png"
    )

    # === 6. Analyse des résidus ===
    # Permet de vérifier la qualité du fit
    y_pred = predict_fn_xraw(X_raw)
    plot_residuals(y_raw, y_pred, bins=20,
                   title="Residuals – Table 4.17",
                   save_path="../Plot/mlp/table_4_17_residuals.png")
