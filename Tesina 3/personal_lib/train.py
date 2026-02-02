
import pandas as pd


from sklearn.base import clone
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import time


from personal_lib.logger import *
from personal_lib.utils import *
SEED = 42

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

        run_y_true =np.zeros(len(y))
        run_y_pred =np.zeros(len(y))
        run_y_proba=np.zeros(len(y))

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


            run_y_true[test_idx] = y_test_fold
            run_y_pred[test_idx] = y_test_pred
            run_y_proba[test_idx] = y_test_proba
            

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
            loss_curve=fold_loss_curve,
            y_true=run_y_true,
            y_pred=run_y_pred,
            y_proba=run_y_proba
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

