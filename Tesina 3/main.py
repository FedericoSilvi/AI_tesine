from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

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
    return X_train, X_test, y_train, y_test

def pipeline_phase_1(seed : int = 42):

    return Pipeline(
        steps=[
            ("scaler",StandardScaler()),
            ("mlp",MLPClassifier(random_state=seed))
        ]
    )

def main_phase_1 ():
    
    data_path = "DARWIN.csv"

    # Load dataset
    print("Dataset in carimento...")
    X,y = load_darwin_dataset(data_path)
    print("Dataset caricato correttamente")


    # Pipeline creation 
    print("Pipeline in creazione...")
    pipeline = pipeline_phase_1()
    print("Pipeline creata correttamente")
    

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Definiamo un set di metriche
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

    print("Esecuzione Cross-Validation...")
    results = cross_validate(
        pipeline, X, y, 
        cv=outer_cv, 
        scoring=scoring_metrics, 
        return_train_score=False,
        n_jobs=-1
    )

    # Visualizzazione pulita dei risultati
    for metric in scoring_metrics:
        mean_score = results[f'test_{metric}'].mean()
        std_score = results[f'test_{metric}'].std()
        print(f"{metric.capitalize()}: {mean_score:.4f} (+/- {std_score:.4f})")

if __name__ == "__main__":
    main_phase_1()

    