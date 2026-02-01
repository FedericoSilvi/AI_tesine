import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
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
        self.training_times = []
        self.train_scores = []
        self.test_scores = []
        self.loss_curves = []
        self.n_iterations = []
        self.sensitivities = [] # true positive rate
        self.specificities = [] # true negative rate
        self.aucs = [] # area under the curve

    def log_run(self, mlp, train_score, test_score, train_time, sens, spec, auc):
        self.training_times.append(train_time)
        self.train_scores.append(train_score)
        self.test_scores.append(test_score)
        self.loss_curves.append(mlp.loss_curve_)
        self.n_iterations.append(mlp.n_iter_)
        self.sensitivities.append(sens)
        self.specificities.append(spec)
        self.aucs.append(auc)

    def get_summary(self):
        return {
            'mean_train_acc': np.mean(self.train_scores),
            'std_train_acc': np.std(self.train_scores),
            'mean_test_acc': np.mean(self.test_scores),
            'std_test_acc': np.std(self.test_scores),
            'mean_time': np.mean(self.training_times),
            'std_time': np.mean(self.training_times),
            'mean_sens': np.mean(self.sensitivities),
            'std_sens': np.std(self.sensitivities),
            'mean_spec': np.mean(self.specificities),
            'std_spec': np.std(self.specificities),
            'mean_auc': np.mean(self.aucs),
            'std_auc': np.std(self.aucs),
            'mean_iterations': np.mean(self.n_iterations)
        }


# Logger esteso per CV
class CVLogger:
    def __init__(self):
        self.train_scores = []
        self.training_times = []
        self.test_scores = []
        self.sensitivities = []
        self.specificities = []
        self.aucs = []
        self.n_iterations = []
        self.loss_curves = []

    def log_run(self, train_acc, test_acc, sens, spec, auc, train_time, n_iter, loss_curve):
        self.train_scores.append(train_acc)
        self.test_scores.append(test_acc)
        self.sensitivities.append(sens)
        self.specificities.append(spec)
        self.aucs.append(auc)
        self.training_times.append(train_time)
        self.n_iterations.append(n_iter)
        self.loss_curves.append(loss_curve)

    def get_summary(self):
        return {
            'mean_train_acc': np.mean(self.train_scores),
            'std_train_acc': np.std(self.train_scores),
            'mean_test_acc': np.mean(self.test_scores),
            'std_test_acc': np.std(self.test_scores),
            'mean_time': np.mean(self.training_times),
            'std_time': np.std(self.training_times),
            'mean_iterations': np.mean(self.n_iterations),
            'mean_sens': np.mean(self.sensitivities),
            'std_sens': np.std(self.sensitivities),
            'mean_spec': np.mean(self.specificities),
            'std_spec': np.std(self.specificities),
            'mean_auc': np.mean(self.aucs),
            'std_auc': np.std(self.aucs),
            'total_evaluations': len(self.test_scores)*5
        }

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

def load_dataset(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    # Dataset loading
    dataset = pd.read_csv(filepath)

    # Features and class extraction
    features = dataset.iloc[:, 1:-1]
    classes = dataset.iloc[:, -1]

    # Missing values management: median substitution
    classes = classes.map({'P': 1, 'H': 0})

    return features, classes


def train_mlp(X: pd.DataFrame, y: pd.Series,
              n_runs: int = 30):
    """
    Addestramento dell'MLP sul train/test split con 30 run
    """

    logger = MLPLogger()
    test_pred = []
    pipeline = None

    for run in range(n_runs):
        print(f"    Run {run + 1}/{n_runs}...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED + run, stratify=y
        )

        print(f"   Train: {X_train.shape[0]} istanze, Test: {X_test.shape[0]} istanze")

        # Crezione della pipeline con scaler
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(random_state=SEED + run))
            # Creazione MLP con seed dipendente da run ma che garantisce riproducibilità
        ])

        start_time = time.time()
        # Addestramento
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predizioni
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Predizione con probabilità per l'AUC
        y_test_proba = pipeline.predict_proba(X_test)[:, 1] # [:, 1] prende la probabilità della classe "Positivo/Malato"

        # Accuracy
        train_acc = accuracy_score(y_train, y_train_pred)

        test_acc, sens, spec, auc = calculate_metrics(y_test, y_test_pred, y_test_proba)

        logger.log_run(pipeline.named_steps["mlp"], train_acc, test_acc, training_time, sens, spec, auc)
        test_pred.append(y_test_pred)

    return logger, test_pred, pipeline


def train_mlp_with_cv(X: pd.DataFrame, y: pd.Series, n_splits=5, n_runs=30):
    """
    Cross-validation
    """

    logger = CVLogger()

    for run in range(n_runs):
        print(f"    CV Run {run + 1}/{n_runs}...")

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + run)

        fold_train_accs = []
        fold_test_accs = []
        fold_sens = []
        fold_specs = []
        fold_aucs = []
        fold_iters = []
        fold_loss_curve = None

        start_time = time.time()

        # Cross validation manuale per poter accedere a tutti i dati
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(random_state=SEED + run))
            ])

            # Estrazione manuale delle fold
            X_train_fold = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
            X_test_fold = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
            y_train_fold = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            y_test_fold = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]

            # Creazione e addestramento del modello sulle fold individuate
            pipeline.fit(X_train_fold, y_train_fold)

            # Predizioni
            y_train_pred = pipeline.predict(X_train_fold)
            y_test_pred = pipeline.predict(X_test_fold)

            # Predizione con probabilità per l'AUC
            y_test_proba = pipeline.predict_proba(X_test_fold)[:, 1]  # [:, 1] prende la probabilità della classe "Positivo/Malato"

            # Metriche
            train_acc = accuracy_score(y_train_fold, y_train_pred)
            fold_train_accs.append(train_acc)

            test_acc, sens, spec, auc = calculate_metrics(y_test_fold, y_test_pred, y_test_proba)

            fold_test_accs.append(test_acc)
            fold_sens.append(sens)
            fold_specs.append(spec)
            fold_aucs.append(auc)
            fold_iters.append(pipeline.named_steps["mlp"].n_iter_)

            if fold_loss_curve is None: # Salva solo la prima curva di apprendimento cioè quella del primo fold, per farlo controlla se fold_loss_curve è ancora None
                fold_loss_curve = pipeline.named_steps["mlp"].loss_curve_

        training_time = time.time() - start_time

        # Media per questo run
        logger.log_run(
            train_acc=np.mean(fold_train_accs),
            test_acc=np.mean(fold_test_accs),
            sens=np.mean(fold_sens),
            spec=np.mean(fold_specs),
            auc=np.mean(fold_aucs),
            train_time=training_time,
            n_iter=np.mean(fold_iters),
            loss_curve=fold_loss_curve
        )

    return logger


def plot_loss_convergence(logger):
    """
    Visualizza tutte le curve di loss salvate nel logger.
    Gestisce curve di lunghezza diversa calcolando una media.
    """
    curves = logger.loss_curves

    if not curves:
        print("Nessuna curva di loss trovata nel logger.")
        return

    plt.figure(figsize=(12, 7))

    # 1. Plot di tutte le singole curve (1 per fold x 30 run)
    for i, curve in enumerate(curves):
        plt.plot(curve, color='deepskyblue', alpha=0.15, linewidth=1)

    # Calcolo della curva MEDIA
    # Poiché le curve hanno lunghezze diverse (le run finiscono in epoche diverse),
    # dobbiamo uniformarle per calcolare la media.

    # Troviamo la lunghezza massima tra tutte le run
    max_len = max(len(c) for c in curves)

    # Creiamo una matrice piena di NaN
    curves_matrix = np.full((len(curves), max_len), np.nan)

    # Riempiamo la matrice
    for i, curve in enumerate(curves):
        curves_matrix[i, :len(curve)] = curve

    # Calcoliamo la media ignorando i NaN (nanmean)
    # Questo ci dà la loss media istante per istante, finché ci sono run attive
    mean_curve = np.nanmean(curves_matrix, axis=0)

    # Plot della media
    plt.plot(mean_curve, color='navy', linewidth=2.5, label='Loss Media')

    # Grafico
    plt.title(f"Curve di Loss: {len(curves)} Run Totali", fontsize=14, fontweight='bold')
    plt.xlabel("Iterazioni (Epoche)", fontsize=12)
    plt.ylabel("Valore Loss", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)

    min_iter = min(len(c) for c in curves)
    max_iter = max_len
    plt.text(0.7, 0.5,
             f"Min Iterazioni: {min_iter}\nMax Iterazioni: {max_iter}",
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig("darwin_convergence_analysis.png", dpi=300)  # salva immagine
    plt.show()


def main_phase_1():
    print("=" * 70)
    print("FASE 1: ANALISI MLP DEFAULT")
    print("=" * 70)

    print("\nCaricamento dataset...")
    X, y = load_dataset(filepath=FILEPATH)
    print(f"   Dataset: {X.shape[0]} istanze, {X.shape[1]} features")
    print(f"   Distribuzione classi: {np.bincount(y)}")

    # ========== CROSS VALIDATION (METODO PRINCIPALE) ==========
    print("\n" + "=" * 70)
    print("CROSS VALIDATION (30 run × 5 fold = 150 valutazioni)")
    print("=" * 70)
    cv_logger = train_mlp_with_cv(X, y, n_splits=5, n_runs=30)

    print("\nMetriche da Cross-Validation:")
    cv_summary = cv_logger.get_summary()
    print(f"   Train Accuracy: {cv_summary['mean_train_acc']:.4f} ± {cv_summary['std_train_acc']:.4f}")
    print(f"   Test Accuracy:  {cv_summary['mean_test_acc']:.4f} ± {cv_summary['std_test_acc']:.4f}")
    print(f"   Tempo medio:    {cv_summary['mean_time']:.3f} ± {cv_summary['std_time']:.3f} s")
    print(f"   Iterazioni medie: {cv_summary['mean_iterations']:.1f}")
    print(f"   Valutazioni totali: {cv_summary['total_evaluations']}")

    # ========== TRAIN/TEST SPLIT (ANALISI SUPPLEMENTARE) ==========
    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT (per confronto)")
    print("=" * 70)

    print("\nTraining con 30 run...")
    split_logger, test_predictions, final_model = train_mlp(
        X, y, n_runs=30
    )

    print("\nMetriche da Train/Test Split:")
    split_summary = split_logger.get_summary()
    print(f"   Train Accuracy: {split_summary['mean_train_acc']:.4f} ± {split_summary['std_train_acc']:.4f}")
    print(f"   Test Accuracy:  {split_summary['mean_test_acc']:.4f} ± {split_summary['std_test_acc']:.4f}")
    print(f"   Tempo medio:    {split_summary['mean_time']:.3f} ± {split_summary['std_time']:.3f} s")
    print(f"   Iterazioni medie: {split_summary['mean_iterations']:.1f}")

    # ========== REPORT FINALE ==========
    print("\n" + "=" * 70)
    print("REPORT FINALE - FASE 1")
    print("=" * 70)
    print("Configurazione: MLPClassifier(random_state=(42 + run_id))")
    print(f"Dataset: {X.shape[0]} istanze, {X.shape[1]} features")

    print("\n--- RISULTATI CROSS VALIDATION (metodo principale) ---")
    print(f"{'Metrica':<15} | {'Media':<10} | {'Std Dev':<10}")
    print("-" * 45)
    print(f"{'Test Accuracy':<15} | {cv_summary['mean_test_acc']:.4f}     | ± {cv_summary['std_test_acc']:.4f}")
    print(f"{'Sensitivity':<15} | {cv_summary['mean_sens']:.4f}     | ± {cv_summary['std_sens']:.4f}")
    print(f"{'Specificity':<15} | {cv_summary['mean_spec']:.4f}     | ± {cv_summary['std_spec']:.4f}")
    print(f"{'AUC':<15} | {cv_summary['mean_auc']:.4f}     | ± {cv_summary['std_auc']:.4f}")
    print("-" * 45)

    print(f"Tempo medio:     {cv_summary['mean_time']:.3f} s")
    print(f"Iterazioni:      {cv_summary['mean_iterations']:.1f}")
    print(f"Train Accuracy:  {cv_summary['mean_train_acc']:.4f}")
    print(f"Valutazioni:     {cv_summary['total_evaluations']}")

    print("\n--- RISULTATI TRAIN/TEST SPLIT (confronto) ---")
    print(f"{'Metrica':<15} | {'Media':<10} | {'Std Dev':<10}")
    print("-" * 45)
    print(f"{'Test Accuracy':<15} | {split_summary['mean_test_acc']:.4f}     | ± {split_summary['std_test_acc']:.4f}")
    print(f"{'Sensitivity':<15} | {split_summary['mean_sens']:.4f}     | ± {split_summary['std_sens']:.4f}")
    print(f"{'Specificity':<15} | {split_summary['mean_spec']:.4f}     | ± {split_summary['std_spec']:.4f}")
    print(f"{'AUC':<15} | {split_summary['mean_auc']:.4f}     | ± {split_summary['std_auc']:.4f}")
    print("-" * 45)

    print(f"Tempo medio:     {split_summary['mean_time']:.3f}s")
    print(f"Iterazioni:      {split_summary['mean_iterations']:.1f}")
    print(f"Train Accuracy:   {split_summary['mean_train_acc']:.4f}")

    print("=" * 70)

    return cv_logger, split_logger, final_model


if __name__ == "__main__":
    cv_logger, split_logger, final_model = main_phase_1()
    plot_loss_convergence(cv_logger)