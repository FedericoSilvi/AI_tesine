from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd 
import numpy as np
from typing import Tuple, List, Dict



def load_darwin_dataset(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carica il dataset DARWIN e gestisce:
        - missing values
        - mapping target
    """

    # Dataset loading
    dataset = pd.read_csv(filepath)

    # Features and class extraction
    features = dataset.iloc[:, 1:-1]
    classes = dataset.iloc[:, -1]

    # Missing values management: median substitution
    features = features.fillna(features.median())
    classes = classes.map({'P':1, 'H' :0})

    return (features, classes)

def preprocessing(X : pd.DataFrame) -> pd.DataFrame:

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled

def split(X : pd.DataFrame, y : pd.Series)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y)
    return [X_train, X_test, y_train, y_test]


def main ():
    
    data_path = "DAWIN.csv"

    X,y = load_darwin_dataset(data_path)

    X_proc, y_proc = preprocessing(X,y)

    X_train, X_test, y_train, y_test = split(X_proc, y_proc)

    # Configurazione default 
    mlp_default = MLPClassifier(random_state=42)

    # Esempio con parametri personalizzati completi 
    mlp_custom = MLPClassifier(
        hidden_layer_sizes=(400,200),       # Due layer con 400 e 200 neuroni 
        activation='tanh',                  # Opzioni: 'identity', 'logistic', 'tanh', 'relu'
        solver='adam',                      # Opzioni: 'lbfgs', 'sgd', 'adam'
        alpha=0.001,                        # Penalità L2                        
        batch_size=32,                      # Dimensione minibatch
        learning_rate='adaptive',           # Opzioni: 'constant', 'invscaling', 'adaptive'
        learning_rate_init=0.001,           # Learning rate iniziale     
        max_iter=1000,                      # Numero massimo di iterazioni
        shuffle=True,                       # Shuffle dei campioni ad ogni iterazione 
        random_state=42,                    # Per riproducibilità
        early_stopping=True,                # Uso validation set per early stopping 
        validation_fraction=0.15,           # Frazioni di dati per la validation 
        n_iter_no_change=10,                # Numero iterazioni senza miglioramento 
        verbose=True                        # Stampa messaggi di progresso 
    )