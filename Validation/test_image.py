# Uniquement pour tester !
import numpy as np

# === Vecteur d'entrée à prédire ===
# Ici, on représente un échantillon du langage des signes sous forme
# de 42 valeurs (features). Les retours à la ligne sont supprimés.
VECTOR = """0.0,0.0,-0.5412844036697247,0.0679886685552408,-0.963302752293578,
0.2322946175637393,-0.8807339449541285,0.3711048158640226,-0.5321100917431193,
0.4249291784702549,-0.8165137614678899,0.5127478753541076,-0.9357798165137616,
0.7082152974504249,-0.9724770642201837,0.830028328611898,-1.0,0.9376770538243626,
-0.4678899082568807,0.5382436260623229,-0.5963302752293578,0.7507082152974505,
-0.6697247706422018,0.8838526912181303,-0.7247706422018348,1.0,-0.1376146788990825,
0.509915014164306,-0.2477064220183486,0.7195467422096318,-0.3394495412844037,
0.8526912181303116,-0.4128440366972477,0.9688385269121812,0.2018348623853211,
0.4419263456090651,0.128440366972477,0.603399433427762,0.036697247706422,
0.7082152974504249,-0.0642201834862385,0.8073654390934845""".replace("\n","")

# === Fonctions d'activation nécessaires pour le MLP ===
def relu(z): return np.maximum(0.0, z)
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z); return e/np.sum(e, axis=1, keepdims=True)

# === Propagation avant ===
def forward(X, W, B, acts=("relu","relu","softmax")):
    """
    Passe avant dans le réseau de neurones.
    - X : entrée (1 échantillon ou batch)
    - W, B : poids et biais du modèle
    - acts : fonctions d'activation par couche
    Retourne les sorties après la dernière couche.
    """
    A = X
    for l, act in enumerate(acts):
        Z = A @ W[l] + B[l]
        A = relu(Z) if act=="relu" else softmax(Z)
    return A

# === Prédiction de la classe ===
def predict_class(X, W, B):
    P = forward(X, W, B)
    return np.argmax(P, axis=1)

# === Conversion classe → lettre ===
def class_to_letter(c):
    """
    Convertit l'index de classe (0=A, 1=B, ...) en lettre correspondante.
    """
    return "ABCDE"[int(c)]  # 0->A, 1->B, …

# === Chargement du modèle entraîné ===
# Les poids W0, W1, W2 et biais B0, B1, B2 sont stockés dans un fichier .npz-
d = np.load("../Outils/mlp_signs_weights.npz")
W = [d["W0"], d["W1"], d["W2"]]
B = [d["B0"], d["B1"], d["B2"]]

# === Conversion du vecteur en array numpy ===
x = np.fromstring(VECTOR, sep=",", dtype=float)

# Vérification de la dimension attendue
if x.size != 42:
    raise ValueError(f"Il faut 42 valeurs, reçu {x.size}.")


# Mise en forme pour correspondre au format attendu par le réseau (1 ligne, 42 colonnes)
x = x.reshape(1, 42)

# === Prédiction ===
pred = predict_class(x, W, B)[0]
print(f"Classe prédite: {pred}  →  Lettre: {class_to_letter(pred)}")
