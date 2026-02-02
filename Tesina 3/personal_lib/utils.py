import pandas as pd
import pickle
import numpy as np

from typing import Tuple

from sklearn.neural_network import MLPClassifier
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

def get_mlp_config(scenario : str ="architecture",SEED : int =42):

    mlp_configs =[]

    if scenario =="architecture":
        hidden_layers = [(200,), (400,), (600,),(400,200),(600,300),(800,400),(400,200,100),(600,300,150)]
        activations = ['identity', 'logistic', 'tanh', 'relu']

        for arch in hidden_layers:
            for act in activations:
                mlp = MLPClassifier(
                    hidden_layer_sizes=arch,
                    activation=act,
                    random_state=SEED
                )
                mlp_configs.append(mlp)

    elif scenario =="learning_rate":
        learning_rates_init = [0.0001, 0.001, 0.01, 0.1]
        learning_rates_policy = ['constant','invscaling','adaptive']
        solvers = ['adam','sgd','lbfgs']
        batch_sizes = [16,32,64]

        for init in learning_rates_init:
            for policy in learning_rates_policy:
                for sol in solvers:
                    for size in batch_sizes:
                        mlp=MLPClassifier(
                            learning_rate_init=init,
                            learning_rate=policy,
                            solver=sol,
                            batch_size=size,
                            random_state=SEED
                        )
                        mlp_configs.append(mlp)
    elif scenario =="regulation":
        alphas = [0.0001, 0.001, 0.01, 0.1, 0.5]
        validation_split = [0.1,0.15,0.2]
        N_iter_no_change = [5,10,20]

        for alpha in alphas:
            for val_frac in validation_split:
                for iter_no_change in N_iter_no_change:
                    mlp=MLPClassifier(
                        early_stopping=True,
                        alpha=alpha,
                        validation_fraction=val_frac,
                        n_iter_no_change=iter_no_change
                    )
                    mlp_configs.append(mlp)

    return mlp_configs