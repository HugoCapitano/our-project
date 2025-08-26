import numpy as np

# ---------- Activations ----------

# Sigmoïde : sortie dans [0,1]
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def d_sigmoid(a): return a * (1.0 - a)

# Tangente hyperbolique : sortie dans [-1,1]
def tanh(z): return np.tanh(z)
def d_tanh(a): return 1.0 - a**2

# ReLU : max(0, z)
def relu(z): return np.maximum(0.0, z)
def d_relu(a): return (a > 0).astype(a.dtype)

# Linéaire (identité)
def linear(z): return z
def d_linear(a): return np.ones_like(a)

# Softmax : normalisation exponentielle pour classification multi-classes
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True) # stabilité numérique
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

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
def one_hot(y, num_classes):
    Y = np.zeros((y.shape[0], num_classes))
    Y[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return Y

# Cross-entropy (classification)
def loss_ce(y_onehot, y_proba):
    eps = 1e-12
    return -np.mean(np.sum(y_onehot * np.log(y_proba + eps), axis=1))

# Erreur quadratique moyenne (régression)
def loss_mse(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred)**2)

# ---------- Init ----------
def init_params(layers, activations, seed=42):
    """
    Initialise les poids et biais pour un réseau MLP.
    layers : liste [n_in, h1, ..., n_out]
    activations : liste des fonctions d'activation, longueur = len(layers)-1
    Retour : (W, b) où :
        W : liste des matrices de poids
        b : liste des vecteurs de biais
    """
    assert len(activations) == len(layers) - 1
    rng = np.random.default_rng(seed)
    W, b = [], []
    for l in range(len(layers)-1):
        n_in, n_out = layers[l], layers[l+1]
        act = activations[l]
        # He pour ReLU, Xavier sinon
        if act == "relu":
            std = np.sqrt(2.0 / n_in)
        else:
            std = np.sqrt(1.0 / (n_in + n_out))
        W.append(rng.normal(0.0, std, size=(n_in, n_out)))
        b.append(np.zeros((1, n_out)))
    return W, b

# ---------- Propagation avant ----------
def forward(X, W, b, activations):
    """
    Effectue la propagation avant à travers toutes les couches.
    Retourne :
        Z_list : liste des sommes pondérées
        A_list : liste des activations (A_list[0] = X)
        A : sortie finale du réseau
    """
    A = X
    Z_list, A_list = [], [A]
    for l, act_name in enumerate(activations):
        Z = A @ W[l] + b[l]
        f, _ = ACT[act_name]
        A = f(Z)
        Z_list.append(Z); A_list.append(A)
    return Z_list, A_list, A  # A = sortie

# ---------- Rétropropagation et mise à jour ----------
def backward_classification(Xb, yb_onehot, W, b, activations, reg_l2=0.0):
    """
    Softmax + cross-entropy en sortie.
    Retourne gradients dW, db.
    """
    Zs, As, Aout = forward(Xb, W, b, activations)
    m = Xb.shape[0]
    L = len(W)

    # Gradient de sortie pour softmax + CE : Aout - Y
    dZ = (Aout - yb_onehot) / m

    dWs = [None]*L
    dbs = [None]*L

    # Boucle en sens inverse (couches)
    for l in reversed(range(L)):
        A_prev = As[l]
        dW = A_prev.T @ dZ + reg_l2 * W[l]
        db = np.sum(dZ, axis=0, keepdims=True)
        dWs[l] = dW
        dbs[l] = db

        if l > 0:
            dA_prev = dZ @ W[l].T
            _, dphi = ACT[activations[l-1]]
            dZ = dA_prev * dphi(As[l])

    return dWs, dbs, Aout

def backward_regression(Xb, yb, W, b, activations, reg_l2=0.0):
    """
    Régression: perte MSE, dernière activation linéaire en général.
    """
    Zs, As, Aout = forward(Xb, W, b, activations)
    m = Xb.shape[0]
    L = len(W)

    # Gradient de sortie pour MSE : (Yhat - Y) / m
    dZ = (Aout - yb) / m

    dWs = [None]*L
    dbs = [None]*L

    for l in reversed(range(L)):
        A_prev = As[l]
        # Appliquer dérivée si pas la dernière couche linéaire
        if not (l == L-1 and activations[l] == "linear"):
            _, dphi = ACT[activations[l]]
            dZ = dZ * dphi(As[l+1])

        dW = A_prev.T @ dZ + reg_l2 * W[l]
        db = np.sum(dZ, axis=0, keepdims=True)
        dWs[l] = dW
        dbs[l] = db

        if l > 0:
            dZ = dZ @ W[l].T

    return dWs, dbs, Aout

# Mise à jour des paramètres (descente de gradient)
def update_params(W, b, dWs, dbs, lr):
    for l in range(len(W)):
        W[l] -= lr * dWs[l]
        b[l] -= lr * dbs[l]

# ---------- Entraînement et prédiction ----------
def fit_classification(X, y, layers, activations,
                       lr=0.05, epochs=300, batch_size=None, reg_l2=0.0,
                       seed=42, verbose=False):
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


def fit_regression(X, y, layers, activations,
                   lr=0.01, epochs=1000, batch_size=None, reg_l2=0.0, seed=42, verbose=False):
    """
    Entraînement d'un MLP pour la régression.
    Retourne :
        W, b : poids et biais appris
        losses : liste des pertes à chaque époque
    """
    W, b = init_params(layers, activations, seed=seed)
    m = X.shape[0]
    if batch_size is None: batch_size = m
    rng = np.random.default_rng(seed)

    losses = []
    for ep in range(epochs):
        idx = np.arange(m); rng.shuffle(idx)
        Xs, ys = X[idx], y[idx]

        total_loss = 0.0
        for s in range(0, m, batch_size):
            e = s + batch_size
            Xb, yb = Xs[s:e], ys[s:e]
            dWs, dbs, Yhat = backward_regression(Xb, yb, W, b, activations, reg_l2=reg_l2)
            update_params(W, b, dWs, dbs, lr)
            total_loss += loss_mse(yb, Yhat) * Xb.shape[0]

        losses.append(total_loss / m)
        if verbose and (ep % max(1, epochs//10) == 0 or ep == epochs-1):
            print(f"[{ep+1}/{epochs}] loss={losses[-1]:.4f}")

    return W, b, losses

# Prédiction des probabilités (classification)
def predict_proba(X, W, b, activations):
    _, _, A = forward(X, W, b, activations)
    return A

# Prédiction des classes (classification)
def predict_class(X, W, b, activations):
    P = predict_proba(X, W, b, activations)
    return np.argmax(P, axis=1)

# Prédiction (régression)
def predict_regression(X, W, b, activations):
    _, _, Y = forward(X, W, b, activations)
    return Y
