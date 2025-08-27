import numpy as np

# ---------- Activations ----------

# Sigmoïde : sortie dans [0,1]
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z)) # Transforme n'importe quelle valeur (réel z) en une valeur entre 0 et 1
def d_sigmoid(a): return a * (1.0 - a) # Dérivée de la sigmoïde en fonction de la sortie a=sigmoid(z)

# Tangente hyperbolique : sortie dans [-1,1]
def tanh(z): return np.tanh(z) # Transforme n'importe quelle valeur (réel z) en une valeur entre -1 et 1
def d_tanh(a): return 1.0 - a**2 # Dérivée de tanh en fonction de la sortie a=tanh(z)

# ReLU : max(0, z)
def relu(z): return np.maximum(0.0, z) # Met tout ce qui est négatif à 0, laisse le reste tel quel
def d_relu(a): return (a > 0).astype(a.dtype) # Dérivée de ReLu : 1 si l'entrée > 0, sinon 0

# Linéaire (identité)
def linear(z): return z # Pas de non-linéarité. La dérivée vaut 1 partout. 
def d_linear(a): return np.ones_like(a) # Dérivée de l'identité = 1

# Softmax : normalisation exponentielle pour classification multi-classes
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True) # stabilité numérique
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)
# Convertit des scores par classe en probabilités (somme à 1)
# On soustrait le max pour éviter les overflow dans exp() (=très grandes valeurs)

# Dictionnaire des fonctions d'activation et de leurs dérivées
# (pour appel dynamique dans le code)
ACT = {
    "sigmoid":  (sigmoid,  d_sigmoid),
    "tanh":     (tanh,     d_tanh),
    "relu":     (relu,     d_relu),
    "linear":   (linear,   d_linear),
    "softmax":  (softmax,  None),  # dérivée gérée directement via cross-entropy
}

# ---------- Pertes ----------

# Encodage one-hot d'un vecteur d'étiquettes
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    Y = np.zeros((y.shape[0], num_classes))
    Y[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return Y
# Transforme un vecteur de labels (0,1,2...) en matrice one-hot
# Ex : [0,2,1] → [[1,0,0]


# Cross-entropy (classification) 
# loss_ce : cross-entropy → mesure l’écart entre probas prédites et vraies classes (classification).
def loss_ce(y_onehot: np.ndarray, y_proba: np.ndarray) -> float:
    eps = 1e-12
    return -np.mean(np.sum(y_onehot * np.log(y_proba + eps), axis=1))

# Erreur quadratique moyenne (régression) → mesure l’écart entre valeurs prédites et vraies (régression).
def loss_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 0.5 * np.mean((y_true - y_pred)**2)

# ---------- Init ----------
def init_params(
    layers: list[int],
    activations: list[str],
    seed: int = 42
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Initialise les poids et biais pour un réseau MLP.
    layers : liste [n_in, h1, ..., n_out] 
    activations : liste des fonctions d'activation, longueur = len(layers)-1
    Retour : (W, b) où :
        W : liste des matrices de poids
        b : liste des vecteurs de biais
    """
    # Layers : décrit les tailles des couches (entrée, cachées, sortie)
    # Activations : décrit les fonctions d'activation entre chaque couche
    assert len(activations) == len(layers) - 1 # Vérifie qu'il y a bien une activation par couche (sauf entrée)
    rng = np.random.default_rng(seed) # Générateur aléatoire reproductible ; on va remplir W (poids) et b (biais).
    W, b = [], []
    for l in range(len(layers)-1): # -1 pour ne pas dépasser la dernière couche (l = 0..L-1)
        n_in, n_out = layers[l], layers[l+1] # n_in = nb neurones en entrée, n_out = nb neurones en sortie
        act = activations[l]
        # He pour ReLU, Xavier sinon
        if act == "relu": # relu pas de nbr négatif
            std = np.sqrt(2.0 / n_in) 
        else:
            std = np.sqrt(1.0 / (n_in + n_out)) # Xavier 
        W.append(rng.normal(0.0, std, size=(n_in, n_out))) # Poids initiaux tirés d'une normale centrée réduite et ajustée
        b.append(np.zeros((1, n_out))) # Biais initialisés à 0
    return W, b  # retourne les listes de poids et biais prets à être utilisés pour l'entraînement

# Pour chaque couche, on crée :
#   W[l] = matrice de poids (n_in, n_out), initialisée aléatoirement.
#       règle He si ReLU, Xavier sinon.
#   b[l] = vecteur de biais (1, n_out), initialisé à zéro.

# ---------- Propagation avant ----------
def forward(
    X: np.ndarray,
    W: list[np.ndarray],
    b: list[np.ndarray],
    activations: list[str]
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Effectue la propagation avant à travers toutes les couches.
    Retourne :
        Z_list (list) : liste des sommes pondérées
        A_list (list) : liste des activations (A_list[0] = X)
        A  : sortie finale du réseau 
    """
    A = X # A = activation (sortie de chaque couche)
    Z_list, A_list = [], [A] # Stocke les Z et A de chaque couche
    for l, act_name in enumerate(activations):
        Z = A @ W[l] + b[l] # Somme pondérée (entrée @ poids + biais)
        f, _ = ACT[act_name] # Récupère la fonction d'activation
        A = f(Z) # Applique l'activation
        Z_list.append(Z); A_list.append(A) # Stocke Z et A
    return Z_list, A_list, A  # A = sortie finale


# X  --init-->  W,b 
# y --one_hot--> Y

# A0 = X
# pour chaque couche l:
#  Zl = Al @ Wl + bl  somme pondérée (entrées × poids + biais)
#  Al = act_l(Zl) activation (ReLU, tanh, softmax...)
 
# sortie = A  sortie finale du réseau
# perte = CE(Y, A)  (ou MSE) erreur (cross-entropy ou MSE)


# ---------- Rétropropagation et mise à jour ----------
def backward_classification(
    Xb: np.ndarray,
    yb_onehot: np.ndarray,
    W: list[np.ndarray],
    b: list[np.ndarray],
    activations: list[str],
    reg_l2: float = 0.0
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Softmax + cross-entropy en sortie.
    Retourne gradients dW, db.
    """
    Zs, As, Aout = forward(Xb, W, b, activations)  # Propagation avant : récupère Z, A, et la sortie finale
    m = Xb.shape[0]                               # Nombre d'échantillons dans le batch
    L = len(W)                                    # Nombre de couches

    # Gradient de sortie pour softmax + cross-entropy : (Aout - Y) / m
    dZ = (Aout - yb_onehot) / m

    dWs = [None]*L    # Liste pour stocker les gradients des poids
    dbs = [None]*L    # Liste pour stocker les gradients des biais

    # Boucle en sens inverse sur les couches (rétropropagation)
    for l in reversed(range(L)):
        A_prev = As[l]    # Activation de la couche précédente
        # Calcul du gradient des poids avec régularisation L2
        dW = A_prev.T @ dZ + reg_l2 * W[l]
        # Calcul du gradient des biais
        db = np.sum(dZ, axis=0, keepdims=True)
        dWs[l] = dW
        dbs[l] = db

        if l > 0:
            # Propagation du gradient vers la couche précédente
            dA_prev = dZ @ W[l].T
            _, dphi = ACT[activations[l-1]]  # Récupère la dérivée de l'activation
            dZ = dA_prev * dphi(As[l])       # Applique la dérivée de l'activation

    return dWs, dbs, Aout  # Retourne les gradients des poids, des biais et la sortie finale

def backward_regression(
    Xb: np.ndarray,
    yb: np.ndarray,
    W: list[np.ndarray],
    b: list[np.ndarray],
    activations: list[str],
    reg_l2: float = 0.0
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Régression: perte MSE, dernière activation linéaire en général.
    Retourne les gradients des poids, des biais et la sortie finale.
    """
    Zs, As, Aout = forward(Xb, W, b, activations)  # Propagation avant : récupère Z, A, et la sortie finale
    m = Xb.shape[0]                                # Nombre d'échantillons dans le batch
    L = len(W)                                     # Nombre de couches

    # Gradient de sortie pour MSE : (Yhat - Y) / m
    dZ = (Aout - yb) / m

    dWs = [None]*L     # Liste pour stocker les gradients des poids
    dbs = [None]*L     # Liste pour stocker les gradients des biais

    # Boucle en sens inverse sur les couches (rétropropagation)
    for l in reversed(range(L)):
        A_prev = As[l]  # Activation de la couche précédente
        # Appliquer la dérivée de l'activation si ce n'est pas la dernière couche linéaire
        if not (l == L-1 and activations[l] == "linear"):
            _, dphi = ACT[activations[l]]      # Récupère la dérivée de l'activation
            dZ = dZ * dphi(As[l+1])           # Applique la dérivée à la sortie de la couche

        # Calcul du gradient des poids avec régularisation L2
        dW = A_prev.T @ dZ + reg_l2 * W[l]
        # Calcul du gradient des biais
        db = np.sum(dZ, axis=0, keepdims=True)
        dWs[l] = dW
        dbs[l] = db

        if l > 0:
            # Propagation du gradient vers la couche précédente
            dZ = dZ @ W[l].T

    return dWs, dbs, Aout  # Retourne les gradients des poids, des biais et la sortie finale

# Mise à jour des paramètres (descente de gradient)
def update_params(
    W: list[np.ndarray],
    b: list[np.ndarray],
    dWs: list[np.ndarray],
    dbs: list[np.ndarray],
    lr: float
) -> None:
    for l in range(len(W)):
        W[l] -= lr * dWs[l]
        b[l] -= lr * dbs[l]

# ---------- Entraînement et prédiction ----------
def fit_classification(
    X: np.ndarray,
    y: np.ndarray,
    layers: list[int],
    activations: list[str],
    lr: float = 0.05,
    epochs: int = 300,
    batch_size: int = None,
    reg_l2: float = 0.0,
    seed: int = 42,
    verbose: bool = False
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[float]]:
    """
    Entraînement d'un MLP pour la classification.
    Retourne :
        W, b : poids et biais appris
        losses : liste des pertes à chaque époque
        accs : liste des accuracy à chaque époque
    """
    W, b = init_params(layers, activations, seed=seed)
    m = X.shape[0]
    if batch_size is None: batch_size = m
    rng = np.random.default_rng(seed)
    Y = one_hot(y, layers[-1])

    losses = []
    accs = []

    for ep in range(epochs):
        idx = np.arange(m); rng.shuffle(idx)
        Xs, Ys = X[idx], Y[idx]

        total_loss = 0.0
        for s in range(0, m, batch_size):
            e = s + batch_size
            Xb, Yb = Xs[s:e], Ys[s:e]
            dWs, dbs, Aout = backward_classification(Xb, Yb, W, b, activations, reg_l2=reg_l2)
            update_params(W, b, dWs, dbs, lr)
            total_loss += loss_ce(Yb, Aout) * Xb.shape[0]

        losses.append(total_loss / m)

        # Calcul accuracy à la fin de l'époque
        y_pred_ep = predict_class(X, W, b, activations)
        accs.append((y_pred_ep == y).mean())

        if verbose and (ep % max(1, epochs//10) == 0 or ep == epochs-1):
            print(f"[{ep+1}/{epochs}] loss={losses[-1]:.4f} acc={accs[-1]:.3f}")

    return W, b, losses, accs        # ← NEW


def update_params(
    W: list[np.ndarray],
    b: list[np.ndarray],
    dWs: list[np.ndarray],
    dbs: list[np.ndarray],
    lr: float
) -> None:
    # Met à jour les poids et biais du réseau avec le gradient calculé
    for l in range(len(W)):
        W[l] -= lr * dWs[l]  # Descente de gradient sur les poids
        b[l] -= lr * dbs[l]  # Descente de gradient sur les biais

# ---------- Entraînement et prédiction ----------
def fit_classification(
    X: np.ndarray,
    y: np.ndarray,
    layers: list[int],
    activations: list[str],
    lr: float = 0.05,
    epochs: int = 300,
    batch_size: int = None,
    reg_l2: float = 0.0,
    seed: int = 42,
    verbose: bool = False
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[float]]:
    """
    Entraînement d'un MLP pour la classification.
    Retourne :
        W, b : poids et biais appris
        losses : liste des pertes à chaque époque
        accs : liste des accuracy à chaque époque
    """
    W, b = init_params(layers, activations, seed=seed)  # Initialisation des poids et biais
    m = X.shape[0]                                      # Nombre d'échantillons
    if batch_size is None: batch_size = m                # Si batch_size non spécifié, on fait du full batch
    rng = np.random.default_rng(seed)                    # Générateur aléatoire pour le shuffle
    Y = one_hot(y, layers[-1])                           # Encodage one-hot des labels

    losses = []  # Liste des pertes (cross-entropy) à chaque époque
    accs = []    # Liste des accuracy à chaque époque

    for ep in range(epochs):                             # Boucle sur les époques
        idx = np.arange(m); rng.shuffle(idx)             # Mélange des indices pour le batch
        Xs, Ys = X[idx], Y[idx]                         # Données mélangées

        total_loss = 0.0
        for s in range(0, m, batch_size):                # Boucle sur les mini-batchs
            e = s + batch_size
            Xb, Yb = Xs[s:e], Ys[s:e]                   # Sélection du batch courant
            dWs, dbs, Aout = backward_classification(Xb, Yb, W, b, activations, reg_l2=reg_l2)  # Rétropropagation
            update_params(W, b, dWs, dbs, lr)            # Mise à jour des paramètres
            total_loss += loss_ce(Yb, Aout) * Xb.shape[0]# Accumulation de la perte pondérée par la taille du batch

        losses.append(total_loss / m)                    # Moyenne de la perte sur l'époque

        # Calcul accuracy à la fin de l'époque
        y_pred_ep = predict_class(X, W, b, activations)  # Prédiction sur tout le dataset
        accs.append((y_pred_ep == y).mean())             # Calcul de l'accuracy

        if verbose and (ep % max(1, epochs//10) == 0 or ep == epochs-1):
            print(f"[{ep+1}/{epochs}] loss={losses[-1]:.4f} acc={accs[-1]:.3f}")

    return W, b, losses, accs        # Retourne les poids, biais, pertes et accuracy


def fit_regression(
    X: np.ndarray,
    y: np.ndarray,
    layers: list[int],
    activations: list[str],
    lr: float = 0.01,
    epochs: int = 1000,
    batch_size: int = None,
    reg_l2: float = 0.0,
    seed: int = 42,
    verbose: bool = False
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """
    Entraînement d'un MLP pour la régression.
    Retourne :
        W, b : poids et biais appris
        losses : liste des pertes à chaque époque
    """
    W, b = init_params(layers, activations, seed=seed)  # Initialisation des poids et biais
    m = X.shape[0]                                      # Nombre d'échantillons
    if batch_size is None: batch_size = m                # Si batch_size non spécifié, on fait du full batch
    rng = np.random.default_rng(seed)                    # Générateur aléatoire pour le shuffle

    losses = []  # Liste des pertes (MSE) à chaque époque
    for ep in range(epochs):                             # Boucle sur les époques
        idx = np.arange(m); rng.shuffle(idx)             # Mélange des indices
        Xs, ys = X[idx], y[idx]                         # Données mélangées

        total_loss = 0.0
        for s in range(0, m, batch_size):                # Boucle sur les mini-batchs
            e = s + batch_size
            Xb, yb = Xs[s:e], ys[s:e]                   # Sélection du batch courant
            dWs, dbs, Yhat = backward_regression(Xb, yb, W, b, activations, reg_l2=reg_l2)  # Rétropropagation
            update_params(W, b, dWs, dbs, lr)            # Mise à jour des paramètres
            total_loss += loss_mse(yb, Yhat) * Xb.shape[0] # Accumulation de la perte pondérée par la taille du batch

        losses.append(total_loss / m)                    # Moyenne de la perte sur l'époque
        if verbose and (ep % max(1, epochs//10) == 0 or ep == epochs-1):
            print(f"[{ep+1}/{epochs}] loss={losses[-1]:.4f}")

    return W, b, losses                                 # Retourne les poids, biais et pertes

# Prédiction des probabilités (classification)
def predict_proba(
    X: np.ndarray,
    W: list[np.ndarray],
    b: list[np.ndarray],
    activations: list[str]
) -> np.ndarray:
    _, _, A = forward(X, W, b, activations)
    return A

# Prédiction des classes (classification)
def predict_class(
    X: np.ndarray,
    W: list[np.ndarray],
    b: list[np.ndarray],
    activations: list[str]
) -> np.ndarray:
    P = predict_proba(X, W, b, activations)
    return np.argmax(P, axis=1)

# Prédiction (régression)
def predict_regression(
    X: np.ndarray,
    W: list[np.ndarray],
    b: list[np.ndarray],
    activations: list[str]
) -> np.ndarray:
    _, _, Y = forward(X, W, b, activations)
    return Y
