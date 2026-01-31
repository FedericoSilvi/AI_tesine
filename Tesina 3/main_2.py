import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve, 
                             auc, classification_report, roc_auc_score)
import time
from typing import Tuple, List, Dict


SEED = 42
FILEPATH = "DARWIN.csv"



class MLPLogger:
    """
    Classe per il Logger dell'MLP base 
    """

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.training_times=[]
        self.train_scores=[]
        self.test_scores=[]
        self.loss_curves=[]
        self.n_iterations=[]

    def log_run(self, mlp, train_score, test_score, train_time):

        self.training_times.append(train_time)
        self.train_scores.append(train_score)
        self.test_scores.append(test_score)
        self.loss_curves.append(mlp.loss_curve_)
        self.n_iterations.append(mlp.n_iter_)

    def get_summary(self):
        return{
            'mean_train_acc' : np.mean(self.train_scores),
            'std_train_acc' : np.std(self.train_scores),
            'mean_test_acc' : np.mean(self.test_scores),
            'std_test_acc' : np.std(self.test_scores),
            'mean_time' : np.mean(self.training_times),
            'std_time' : np.mean(self.training_times),
            'mean_iterations': np.mean(self.n_iterations)
        }




def load_dataset(filepath : str) -> Tuple[pd.DataFrame, pd.Series]:

    
    # Dataset loading
    dataset = pd.read_csv(filepath)

    # Features and class extraction
    features = dataset.iloc[:, 1:-1]
    classes = dataset.iloc[:, -1]

    # Missing values management: median substitution
    features = features.fillna(features.median())
    classes = classes.map({'P':1, 'H' :0})

    return features, classes


def train_mlp(X_train : pd.DataFrame, y_train : pd.Series, X_test : pd.DataFrame, y_test : pd.Series, 
            n_runs : int =30):
    """
    Addestramento dell'MLP sul train/test split con 30 run
    """

    logger = MLPLogger()
    test_pred = []

    for run in range(n_runs):
        print(f"    Run {run+1}/{n_runs}...")

        # Creazione MLP con seed dipendente da run ma che garantisce riproducibilità
        mlp = MLPClassifier(random_state=SEED+run)

        start_time = time.time()
        # Addestramento
        mlp.fit(X_train, y_train)
        training_time = time.time()-start_time

        # Predizioni 
        y_train_pred = mlp.predict(X_train)
        y_test_pred = mlp.predict(X_test)

        #Accuracy
        train_acc = accuracy_score(y_train,y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        logger.log_run(mlp, train_acc, test_acc, training_time)
        test_pred.append(y_test_pred)

    return logger, test_pred, mlp

def train_mlp_with_cv(X : pd.DataFrame, y : pd.Series, n_splits = 5, n_runs = 30):
    """
    Cross-validation 
    """
    # Logger esteso per CV
    class CVLogger:
        def __init__(self):
            self.training_times = []
            self.train_scores = []      # Media per run
            self.test_scores = []       # Media per run
            self.all_train_scores = []  # Tutti i fold
            self.all_test_scores = []   # Tutti i fold
            self.all_loss_curves = []   # Tutte le curve
            self.all_n_iterations = []  # Tutte le iterazioni
            self.n_iterations = []      # Media per run
            self.loss_curves = []       # Curve rappresentative

        def get_summary(self):
            return {
                'mean_train_acc': np.mean(self.train_scores),
                'std_train_acc': np.std(self.train_scores),
                'mean_test_acc': np.mean(self.test_scores),
                'std_test_acc': np.std(self.test_scores),
                'mean_time': np.mean(self.training_times),
                'std_time': np.std(self.training_times),
                'mean_iterations': np.mean(self.n_iterations),
                'total_evaluations': len(self.all_train_scores)  # n_runs × n_splits
            }
    logger = CVLogger()
    
    for run in range(n_runs):
        print(f"    CV Run {run+1}/{n_runs}...")
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + run)
        
        run_train_scores = []
        run_test_scores = []
        run_loss_curves = []
        run_n_iters = []
        
        start_time = time.time()
        
        # Cross validation manuale per poter accedere a tutti i dati 
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):

            # Estrazione manuale delle fold
            X_train_fold = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
            X_test_fold = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
            y_train_fold = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            y_test_fold = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
            
            # Creazione e addestramento del modello sulle fold individuate
            mlp = MLPClassifier(random_state=SEED + run)
            mlp.fit(X_train_fold, y_train_fold)
            
            # Predizioni 
            y_train_pred = mlp.predict(X_train_fold)
            y_test_pred = mlp.predict(X_test_fold)
            
            # Metriche
            train_acc = accuracy_score(y_train_fold, y_train_pred)
            test_acc = accuracy_score(y_test_fold, y_test_pred)
            
            run_train_scores.append(train_acc)
            run_test_scores.append(test_acc)
            run_loss_curves.append(mlp.loss_curve_)
            run_n_iters.append(mlp.n_iter_)
            
            # Salva anche nei risultati completi
            logger.all_train_scores.append(train_acc)
            logger.all_test_scores.append(test_acc)
            logger.all_loss_curves.append(mlp.loss_curve_)
            logger.all_n_iterations.append(mlp.n_iter_)
        
        training_time = time.time() - start_time
        
        # Media per questo run
        logger.training_times.append(training_time)
        logger.train_scores.append(np.mean(run_train_scores))
        logger.test_scores.append(np.mean(run_test_scores))
        logger.n_iterations.append(np.mean(run_n_iters))
        logger.loss_curves.append(run_loss_curves[0])  # Rappresentativa
    
    return logger




def main_phase_1():
    
    print("="*70)
    print("FASE 1: ANALISI MLP DEFAULT")
    print("="*70)
    
    print("\nCaricamento dataset...")
    X, y = load_dataset(filepath=FILEPATH)
    print(f"   Dataset: {X.shape[0]} istanze, {X.shape[1]} features")
    print(f"   Distribuzione classi: {np.bincount(y)}")
    
    print("\nStandardizzazione...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ========== CROSS VALIDATION (METODO PRINCIPALE) ==========
    print("\n" + "="*70)
    print("CROSS VALIDATION (30 run × 5 fold = 150 valutazioni)")
    print("="*70)
    cv_logger = train_mlp_with_cv(X_scaled, y, n_splits=5, n_runs=30)
    
    print("\nMetriche da Cross-Validation:")
    cv_summary = cv_logger.get_summary()
    print(f"   Train Accuracy: {cv_summary['mean_train_acc']:.4f} ± {cv_summary['std_train_acc']:.4f}")
    print(f"   Test Accuracy:  {cv_summary['mean_test_acc']:.4f} ± {cv_summary['std_test_acc']:.4f}")
    print(f"   Tempo medio:    {cv_summary['mean_time']:.3f} ± {cv_summary['std_time']:.3f} s")
    print(f"   Iterazioni medie: {cv_summary['mean_iterations']:.1f}")
    print(f"   Valutazioni totali: {cv_summary['total_evaluations']}")
    
    # ========== TRAIN/TEST SPLIT (ANALISI SUPPLEMENTARE) ==========
    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT (per confronto)")
    print("="*70)
    
    print("\nSplitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} istanze, Test: {X_test.shape[0]} istanze")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining con 30 run...")
    split_logger, test_predictions, final_model = train_mlp(
        X_train_scaled, y_train, X_test_scaled, y_test, n_runs=30
    )
    
    print("\nMetriche da Train/Test Split:")
    split_summary = split_logger.get_summary()
    print(f"   Train Accuracy: {split_summary['mean_train_acc']:.4f} ± {split_summary['std_train_acc']:.4f}")
    print(f"   Test Accuracy:  {split_summary['mean_test_acc']:.4f} ± {split_summary['std_test_acc']:.4f}")
    print(f"   Tempo medio:    {split_summary['mean_time']:.3f} ± {split_summary['std_time']:.3f} s")
    print(f"   Iterazioni medie: {split_summary['mean_iterations']:.1f}")
    
    # ========== REPORT FINALE ==========
    print("\n" + "="*70)
    print("REPORT FINALE - FASE 1")
    print("="*70)
    print("Configurazione: MLPClassifier(random_state=42)")
    print(f"Dataset: {X.shape[0]} istanze, {X.shape[1]} features")
    
    print("\n--- CROSS VALIDATION (metodo principale) ---")
    print(f"Accuracy:        {cv_summary['mean_test_acc']:.4f} ± {cv_summary['std_test_acc']:.4f}")
    print(f"Tempo medio:     {cv_summary['mean_time']:.3f}s")
    print(f"Iterazioni:      {cv_summary['mean_iterations']:.1f}")
    print(f"Valutazioni:     {cv_summary['total_evaluations']}")
    
    print("\n--- TRAIN/TEST SPLIT (confronto) ---")
    print(f"Accuracy:        {split_summary['mean_test_acc']:.4f} ± {split_summary['std_test_acc']:.4f}")
    print(f"Tempo medio:     {split_summary['mean_time']:.3f}s")
    print(f"Iterazioni:      {split_summary['mean_iterations']:.1f}")
    
    print("="*70)
    
    return cv_logger, split_logger, final_model


if __name__ == "__main__":
    main_phase_1()