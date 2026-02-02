import pandas as pd
import pickle
import numpy as np

from typing import Tuple

from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score)



def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: pd.Series) -> Tuple[float, float, float, float]:
    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # true positive rate, true negative rate, false positive rate, false negative rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() # ravel appiattisce la matrice in un array

    # Sensibilità
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Specificità
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUC
    auc = 0.5  # Valore neutro di default
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            pass

    return acc, sensitivity, specificity, auc

def load_dataset(filepath: str, selection : bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    # Dataset loading
    dataset = pd.read_csv(filepath)

    # Features and class extraction
    features = dataset.iloc[:, 1:-1]
    if selection == True:

        with open("best_feat_list", "rb") as f:
            selected_features = pickle.load(f)

        print("Feature dal PSO:")
        print(selected_features)
        features_ind = [ind for ind, val in enumerate(selected_features) if val == 1]
        print("Indici:")
        print(features_ind)
        features = features.iloc[:, features_ind]
        
        
    
    print("Feature selezionate:")
    print(features)
    classes = dataset.iloc[:, -1]

    # Missing values management: median substitution
    classes = classes.map({'P': 1, 'H': 0})

    return features, classes