import pandas as pd
import pickle

from sklearn.base import clone
from sklearn.model_selection import learning_curve
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


def train_mlp_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_runs: int = 30,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 8),
    scoring: str = "accuracy",
    # ---- Accuracy per epoca (warm_start) ----
    compute_epoch_accuracy: bool = True,
    max_epochs: int = 200,
    epoch_mode: str = "first_run_first_fold",  # alternative: "first_run_all_folds", "all_runs_first_fold", "all"
    mlp : MLPClassifier = None 
):
    """
    Cross-validation ripetuta (n_runs) con pipeline definita UNA volta.
    In più calcola learning curves classiche (train/val vs train_size).
    Inoltre (opzionale) calcola accuracy per epoca usando warm_start=True.

    Ritorna:
      - logger (come prima)
      - lc_results: dict learning curve classica
      - epoch_results: dict accuracy-per-epoca (oppure None se disabilitato)
    """

    logger = CVLogger()

    # Pipeline DEFINITA UNA VOLTA (config base)
    if mlp == None :
        base_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(random_state=SEED, warm_start=True, max_iter=1))
        ])
    else:
        base_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("mlp", mlp)
        ])
    
    # Accumulatori per learning curves classiche su tutte le run
    all_lc_train_scores = []
    all_lc_val_scores = []
    lc_train_sizes_ref = None

    # Accumulatori accuracy-per-epoca
    epoch_results = None
    epoch_train_curves = []
    epoch_val_curves = []

    def _should_compute_epoch_curve(run_idx: int, fold_idx: int) -> bool:
        if not compute_epoch_accuracy:
            return False
        if epoch_mode == "first_run_first_fold":
            return (run_idx == 0 and fold_idx == 0)
        if epoch_mode == "first_run_all_folds":
            return (run_idx == 0)
        if epoch_mode == "all_runs_first_fold":
            return (fold_idx == 0)
        if epoch_mode == "all":
            return True
        # fallback
        return (run_idx == 0 and fold_idx == 0)

    for run in range(n_runs):
        print(f"    CV Run {run + 1}/{n_runs}...")

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + run)

        # Pipeline per questa run (seed aggiornato)
        pipeline_for_run = clone(base_pipeline).set_params(mlp__random_state=SEED + run)

        fold_train_accs = []
        fold_test_accs = []
        fold_sens = []
        fold_specs = []
        fold_aucs = []
        fold_iters = []
        fold_loss_curve = None

        start_time = time.time()

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            # Pipeline unica con warm_start
            fold_pipe = clone(pipeline_for_run).set_params(
                mlp__warm_start=True,
                mlp__max_iter=1
            )

            X_train_fold = X.iloc[train_idx]
            X_test_fold  = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold  = y.iloc[test_idx]

            # Decidi se salvare accuracy per epoca per questo fold
            save_epoch_curves = _should_compute_epoch_curve(run, fold_idx)
            
            if save_epoch_curves:
                train_acc_epoch = []
                val_acc_epoch = []

            # Training epoca per epoca
            for epoch in range(max_epochs):
                fold_pipe.fit(X_train_fold, y_train_fold)

                if save_epoch_curves:
                    y_tr = fold_pipe.predict(X_train_fold)
                    y_va = fold_pipe.predict(X_test_fold)
                    train_acc_epoch.append(accuracy_score(y_train_fold, y_tr))
                    val_acc_epoch.append(accuracy_score(y_test_fold, y_va))

            # Salva le curve se necessario
            if save_epoch_curves:
                epoch_train_curves.append(train_acc_epoch)
                epoch_val_curves.append(val_acc_epoch)

            # Metriche finali (dopo tutte le epoche)
            y_train_pred = fold_pipe.predict(X_train_fold)
            y_test_pred  = fold_pipe.predict(X_test_fold)
            y_test_proba = fold_pipe.predict_proba(X_test_fold)[:, 1]

            train_acc = accuracy_score(y_train_fold, y_train_pred)
            fold_train_accs.append(train_acc)

            test_acc, sens, spec, aucv = calculate_metrics(y_test_fold, y_test_pred, y_test_proba)
            fold_test_accs.append(test_acc)
            fold_sens.append(sens)
            fold_specs.append(spec)
            fold_aucs.append(aucv)

            mlp_est = fold_pipe.named_steps["mlp"]
            fold_iters.append(mlp_est.n_iter_)

            if fold_loss_curve is None and hasattr(mlp_est, "loss_curve_"):
                fold_loss_curve = mlp_est.loss_curve_

        training_time = time.time() - start_time

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

        # ===== Learning curve "classica" per questa run =====
        # Qui dobbiamo usare una pipeline che fa fit completo, non epoca per epoca
        lc_pipeline = clone(pipeline_for_run).set_params(
            mlp__warm_start=False,
            mlp__max_iter=max_epochs
        )
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator=lc_pipeline,
            X=X,
            y=y,
            cv=cv,
            train_sizes=train_sizes,
            scoring=scoring,
            n_jobs=-1,
            shuffle=False
        )

        lc_train_sizes_ref = train_sizes_abs
        all_lc_train_scores.append(train_scores)
        all_lc_val_scores.append(val_scores)

    # ===== Aggregazione learning curves classiche =====
    all_lc_train_scores = np.array(all_lc_train_scores)
    all_lc_val_scores = np.array(all_lc_val_scores)

    mean_train = all_lc_train_scores.mean(axis=(0, 2))
    std_train  = all_lc_train_scores.std(axis=(0, 2))
    mean_val   = all_lc_val_scores.mean(axis=(0, 2))
    std_val    = all_lc_val_scores.std(axis=(0, 2))

    lc_results = {
        "train_sizes": lc_train_sizes_ref,
        "mean_train_score": mean_train,
        "std_train_score": std_train,
        "mean_val_score": mean_val,
        "std_val_score": std_val,
        "scoring": scoring,
        "n_runs": n_runs,
        "n_splits": n_splits
    }

    # ===== Aggregazione accuracy-per-epoca =====
    if compute_epoch_accuracy and len(epoch_train_curves) > 0:
        epoch_train_curves = np.array(epoch_train_curves)  # shape: (k, max_epochs)
        epoch_val_curves = np.array(epoch_val_curves)

        epoch_results = {
            "epochs": np.arange(1, max_epochs + 1),
            "mean_train_acc": epoch_train_curves.mean(axis=0),
            "std_train_acc": epoch_train_curves.std(axis=0),
            "mean_val_acc": epoch_val_curves.mean(axis=0),
            "std_val_acc": epoch_val_curves.std(axis=0),
            "mode": epoch_mode,
            "curves_count": int(epoch_train_curves.shape[0])
        }

    return logger, lc_results, epoch_results

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

def plot_learning_curves(lc_results, save_path="darwin_learning_curves.png"):
    train_sizes = lc_results["train_sizes"]
    mean_train = lc_results["mean_train_score"]
    std_train  = lc_results["std_train_score"]
    mean_val   = lc_results["mean_val_score"]
    std_val    = lc_results["std_val_score"]

    plt.figure(figsize=(10, 6))

    plt.plot(train_sizes, mean_train, marker="o", label="Train score")
    plt.fill_between(train_sizes, mean_train - std_train, mean_train + std_train, alpha=0.2)

    plt.plot(train_sizes, mean_val, marker="o", label="Validation score")
    plt.fill_between(train_sizes, mean_val - std_val, mean_val + std_val, alpha=0.2)

    plt.title("Learning Curves (Accuracy)")
    plt.xlabel("Numero di esempi di training")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_epoch_accuracy(epoch_results, save_path="accuracy_per_epoch.png"):
    if epoch_results is None:
        print("epoch_results è None (compute_epoch_accuracy=False o nessuna curva calcolata).")
        return

    e = epoch_results["epochs"]
    mt = epoch_results["mean_train_acc"]
    st = epoch_results["std_train_acc"]
    mv = epoch_results["mean_val_acc"]
    sv = epoch_results["std_val_acc"]

    plt.figure(figsize=(10, 6))
    plt.plot(e, mt, label="Train accuracy")
    plt.fill_between(e, mt - st, mt + st, alpha=0.2)

    plt.plot(e, mv, label="Validation accuracy")
    plt.fill_between(e, mv - sv, mv + sv, alpha=0.2)

    plt.xlabel("Epoche")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy per epoca (warm_start) | mode={epoch_results['mode']} | curves={epoch_results['curves_count']}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Questa funzione plotta una matrice di confusione professionale usando solo Matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Settiamo i tick
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Classe Reale',
           xlabel='Classe Predetta')

    # Ruotiamo le etichette sull'asse x
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Inseriamo i numeri all'interno dei quadrati
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.show()

def main_phase_1(feature_selection : bool = False):
    print("=" * 70)
    print("FASE 1: ANALISI MLP DEFAULT")
    print("=" * 70)

    print("\nCaricamento dataset...")
    X, y = load_dataset(filepath=FILEPATH,selection=feature_selection)
    print(f"   Dataset: {X.shape[0]} istanze, {X.shape[1]} features")
    print(f"   Distribuzione classi: {np.bincount(y)}")


    print("Train/Test split del dataset caricato...")
    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=SEED, stratify=y)


    # ========== CROSS VALIDATION (METODO PRINCIPALE) ==========
    print("\n" + "=" * 70)
    print("CROSS VALIDATION (30 run × 5 fold = 150 valutazioni)")
    print("=" * 70)
    cv_logger, lc_results, epoch_results = train_mlp_with_cv(X_train, y_train, n_splits=5, n_runs=30, epoch_mode="all")

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

    """print("\nMetriche da Train/Test Split:")
    split_summary = split_logger.get_summary()
    print(f"   Train Accuracy: {split_summary['mean_train_acc']:.4f} ± {split_summary['std_train_acc']:.4f}")
    print(f"   Test Accuracy:  {split_summary['mean_test_acc']:.4f} ± {split_summary['std_test_acc']:.4f}")
    print(f"   Tempo medio:    {split_summary['mean_time']:.3f} ± {split_summary['std_time']:.3f} s")
    print(f"   Iterazioni medie: {split_summary['mean_iterations']:.1f}") """

    # ========== REPORT FINALE ==========
    print("\n" + "=" * 70)
    print("REPORT FINALE - FASE 1")
    print("=" * 70)
    print("Configurazione: MLPClassifier(random_state=(42 + run_id))")
    print(f"Dataset: {X.shape[0]} istanze, {X.shape[1]} features")

    print("\n--- RISULTATI CROSS VALIDATION (metodo principale) ---")
    print(f"{'Metrica':<15} | {'Media':<10} | {'Std Dev':<10}")
    print("-" * 45)
    print(f"{'Validation Accuracy':<15} | {cv_summary['mean_test_acc']:.4f}     | ± {cv_summary['std_test_acc']:.4f}")
    print(f"{'Sensitivity':<15} | {cv_summary['mean_sens']:.4f}     | ± {cv_summary['std_sens']:.4f}")
    print(f"{'Specificity':<15} | {cv_summary['mean_spec']:.4f}     | ± {cv_summary['std_spec']:.4f}")
    print(f"{'AUC':<15} | {cv_summary['mean_auc']:.4f}     | ± {cv_summary['std_auc']:.4f}")
    print("-" * 45)

    print(f"Tempo medio:     {cv_summary['mean_time']:.3f} s")
    print(f"Iterazioni:      {cv_summary['mean_iterations']:.1f}")
    print(f"Validation Accuracy:  {cv_summary['mean_train_acc']:.4f}")
    print(f"Valutazioni:     {cv_summary['total_evaluations']}")

    """ print("\n--- RISULTATI TRAIN/TEST SPLIT (confronto) ---")
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

    print("=" * 70) """

    


    return cv_logger, split_logger, final_model, lc_results, epoch_results


if __name__ == "__main__":

    MODE = "test"              # train, test
    EXPERIMENT = "default"      # default, scenario 1, scenario 2, scenario 3
    SELECTION = False           # False (no selezione), True (selezione)

    LOGGER_FILE = "pickles/"+EXPERIMENT+"_cv_logger"
    MODEL_FILE = "pickles/"+EXPERIMENT+"_model"
    LC_FILE = "pickles/"+EXPERIMENT+"_lc"
    EPOCH_FILE = "pickles/"+EXPERIMENT+"_epoch"
    X_TEST_FILE ="pickles/"+EXPERIMENT+"_x_test"
    Y_TEST_FILE = "pickles/"+EXPERIMENT+"y_test"

    if SELECTION == False:
        LOGGER_FILE+="_no_sel"
        MODEL_FILE+="_no_sel"
        LC_FILE+="_no_sel"
        EPOCH_FILE+="_no_sel"
    else:
        LOGGER_FILE+="_with_sel"
        MODEL_FILE+="_with_sel"
        LC_FILE+="_with_sel"
        EPOCH_FILE+="_with_sel"

    if MODE == "train":
        if EXPERIMENT == "default":
            cv_logger, split_logger, final_model, lc_results, epoch_results= main_phase_1(SELECTION)
        
        with open(LOGGER_FILE, "wb") as f:
            pickle.dump(cv_logger, f)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(final_model, f)
        with open(LC_FILE, "wb") as f:
            pickle.dump(lc_results, f)
        with open(EPOCH_FILE, "wb") as f:
            pickle.dump(epoch_results, f)
       
        
    elif MODE == "test":

        with open(LOGGER_FILE, "rb") as f:
            cv_logger = pickle.load(f)
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        with open(LC_FILE, "rb") as f:
            lc_results = pickle.load(f)
        with open(EPOCH_FILE, "rb") as f:
            epoch_results = pickle.load(f)
       

        plot_loss_convergence(cv_logger)
        plot_learning_curves(lc_results)
        plot_epoch_accuracy(epoch_results)

    """  print("\n--- PERFORMANCE SUL TEST SET ---")
        y_pred = model.predict(X_test)
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        plot_confusion_matrix(y_test, y_pred, ['P','H']) """
