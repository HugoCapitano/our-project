import numpy as np
import pandas as pd
from pathlib import Path

def build_signs_datasets(
    src: str = "../Datas/LangageDesSignes/data_formatted.csv",
    dst_dir: str = "../Datas/LangageDesSignes",
    per_class_learn: int = 50,
    per_class_val: int = 10,
    seed: int = 42
) -> None:
    """
    Construit deux fichiers CSV pour l'apprentissage et la validation
    à partir du dataset de langage des signes :
      - learning_dataset.csv
      - validation_dataset.csv

    Chaque fichier contient un échantillonnage équilibré par classe.

    Paramètres :
    ------------
    src : str
        Chemin vers le fichier source contenant toutes les données formatées.
    dst_dir : str
        Répertoire où seront sauvegardés les fichiers de sortie.
    per_class_learn : int
        Nombre d'échantillons par classe dans le dataset d'apprentissage.
    per_class_val : int
        Nombre d'échantillons par classe dans le dataset de validation.
    seed : int
        Graine aléatoire pour rendre l'échantillonnage reproductible.
    """
    # --- Chargement des données depuis le fichier source ---
    X, y = [], []
    data_idx = 0
    with open(src, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):

            # Sauter les lignes vides
            if not row or all(cell.strip() == "" for cell in row):
                continue

            # Lecture des 42 premières colonnes comme features
            try:
                feats = [float(v) for v in row[:42]]
            except ValueError:

                continue

            # Détermination de la classe à partir de l'index (1 à 5)
            # Hypothèse : 60 échantillons par classe dans le dataset source
            X.append(feats)
            cls = (data_idx // 60) + 1
            y.append(cls)
            data_idx += 1

    # Conversion en tableaux NumPy
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    # --- Création des datasets apprentissage / validation ---
    rng = np.random.default_rng(seed)
    learn_rows = [] # indices pour l'apprentissage
    val_rows = []   # indices pour la validation

    for cls in np.unique(y):
        # Indices des échantillons appartenant à cette classe
        idx_cls = np.where(y == cls)[0]
        rng.shuffle(idx_cls)

        # Sélection pour apprentissage et validation
        learn_idx = idx_cls[:per_class_learn]
        val_idx   = idx_cls[per_class_learn:per_class_learn+per_class_val]

        learn_rows.extend(learn_idx)
        val_rows.extend(val_idx)

    # --- Sauvegarde des datasets ---
    df_learn = pd.DataFrame(np.c_[y[learn_rows], X[learn_rows]])
    df_val   = pd.DataFrame(np.c_[y[val_rows], X[val_rows]])

    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    df_learn.to_csv(Path(dst_dir)/"learning_dataset.csv", index=False, header=False)
    df_val.to_csv(Path(dst_dir)/"validation_dataset.csv", index=False, header=False)

    print(f"[OK] {len(df_learn)} lignes écrites dans learning_dataset.csv")
    print(f"[OK] {len(df_val)} lignes écrites dans validation_dataset.csv")

if __name__ == "__main__":
    import csv
    build_signs_datasets()
