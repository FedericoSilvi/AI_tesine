
from typing import Dict
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from sklearn.metrics import (confusion_matrix, roc_curve,auc)


#============ DEFAULT SCENARIO PLOTS ============
def plot_loss_convergence(logger,scenario = "_",save_path="_convergence_analysis.png"):
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
    plt.savefig("Immagini/"+scenario+"/"+save_path, dpi=300)  # salva immagine
    plt.show()

def plot_learning_curves(lc_results, scenario ="_", save_path="_darwin_learning_curves.png"):
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
    plt.savefig("Immagini/"+scenario+"/"+save_path, dpi=300)
    plt.show()

def plot_epoch_accuracy(epoch_results, scenario ="_", save_path="_accuracy_per_epoch.png"):
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
    plt.savefig("Immagini/"+scenario+"/"+save_path, dpi=300)
    plt.show()

def plot_cv_confusion_matrix(y_true_list, y_pred_list, classes, scenario ="_",save_path="_total_confusion_matrix",title='Confusion Matrix (CV Total)', cmap=plt.cm.Blues):
    """
    Restituisce la confusion matrix di tutte le predizioni fatte sulla validation del CV
    """
    # 1. Trasformazione forzata in liste piatte
    def flatten(input_list):
        flat = []
        for item in input_list:
            if isinstance(item, (pd.Series, np.ndarray)):
                flat.extend(item.tolist())
            elif isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return np.array(flat)

    y_true_flat = flatten(y_true_list)
    y_pred_flat = flatten(y_pred_list)

    # Verifica coerenza dopo l'appiattimento
    if len(y_true_flat) != len(y_pred_flat):
        raise ValueError(f"Errore: y_true ha {len(y_true_flat)} campioni, y_pred ne ha {len(y_pred_flat)}")

    # 2. Calcolo della matrice
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Classe Reale',
           xlabel='Classe Predetta')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 4. Inserimento numeri e percentuali
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_label = f"{cm[i, j]}\n({(100 * cm[i, j] / np.sum(cm)):.1f}%)"
            ax.text(j, i, text_label,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig("Immagini/"+scenario+"/"+save_path, dpi=300)
    plt.show()


def plot_cv_roc_curve(y_true_list, y_proba_list, scenario ="_",save_path="_roc_curves",title='ROC Curve - Media su 30 Run'):
    """
    Grafica:
    - una curva ROC per ogni run (30 curve con colore leggero)
    - una curva ROC media tra le 30 (colore più marcato)
    """
    plt.figure(figsize=(10, 8))
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Ora y_true_list ha lunghezza 30, non 150
    for i, (y_true, y_proba) in enumerate(zip(y_true_list, y_proba_list)):
        # Calcoliamo la curva per la run i-esima
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        
        # Interpolazione per poter fare la media dopo
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Plot delle singole 30 run (grigio leggero)
        plt.plot(fpr, tpr, lw=1, alpha=0.15, color='gray')

    # Diagonale del caso
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # Media e Deviazione Standard
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot della media (blu)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=fr'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
             lw=2.5, alpha=1)

    # Area di deviazione standard
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=.1,
                     label=fr'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("Immagini/"+scenario+"/"+save_path, dpi=300)
    plt.show()


def plot_accuracy_distribution(cv_logger, scenario ="_",save_path ="_accuracy_distribution",title="Distribuzione dell'Accuracy"):
    scores = cv_logger.test_scores
    
    plt.figure(figsize=(8, 6))
    
    # Istogramma
    plt.hist(scores, bins=10, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    
    # Calcolo KDE
    kde = gaussian_kde(scores)
    x_range = np.linspace(min(scores) - 0.05, max(scores) + 0.05, 100)
    plt.plot(x_range, kde(x_range), color='navy', lw=2, label='Densità (KDE)')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Accuracy")
    plt.ylabel("Frequenza")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("Immagini/"+scenario+"/"+save_path, dpi=300)

    plt.show()

def plot_accuracy_boxplot(cv_logger, scenario ="_",save_path ="_accuracy_boxplot", title="Box Plot Accuracy"):
    scores = cv_logger.test_scores
    
    plt.figure(figsize=(6, 6))
    
    # Creazione del boxplot
    plt.boxplot(scores, vert=True, patch_artist=True, 
                boxprops=dict(facecolor="lightblue", color="navy"),
                medianprops=dict(color="red", lw=2))
    
    plt.title(title, fontsize=14, fontweight='bold')
    
    
    plt.xticks([1], ["Configurazione di Default"])
    
    plt.ylabel("Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig("Immagini/"+scenario+"/"+save_path, dpi=300)

    plt.show()

def plot_cv_prediction_stability(y_true_list, y_pred_list, n_runs=30, scenario ="_",save_path="_prediction_stability", title="Analisi Stabilità delle Predizioni (CV)"):

    """
    y_true_list: lista di liste/array (tutti i fold di tutte le run)
    y_pred_list: lista di liste/array (tutte le predizioni di tutte le run)
    n_runs: numero di run completate
    """
    
    # 1. Appiattimento forzato (Flattening)
    def flatten(input_list):
        flat = []
        for item in input_list:
            if isinstance(item, (pd.Series, np.ndarray)):
                flat.extend(item.tolist())
            elif isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return np.array(flat)

    y_true_all = flatten(y_true_list)
    y_pred_all = flatten(y_pred_list)

    # 2. Ricostruzione della matrice (n_runs x n_samples)
    # Calcoliamo quanti campioni ci sono in una singola run (il dataset intero)
    n_samples_total = len(y_true_all) // n_runs
    
    # Ridimensioniamo le predizioni in una matrice: ogni riga è una run completa
    try:
        preds_matrix = y_pred_all.reshape(n_runs, n_samples_total)
        # Prendiamo un solo set di y_true (le etichette sono le stesse per ogni run)
        y_true_single = y_true_all[:n_samples_total]
    except ValueError:
        print("Errore: Il numero totale di campioni non è divisibile per n_runs.")
        return

    # 3. Calcolo della stabilità e incertezza
    stability_mean = np.mean(preds_matrix, axis=0)
    uncertainty = 4 * stability_mean * (1 - stability_mean) # 0 a stabili, 1 a massima incertezza (0.5)

    # 4. Ordinamento per visualizzazione
    sorted_idx = np.argsort(stability_mean)
    sorted_stability = stability_mean[sorted_idx]
    sorted_actual = y_true_single[sorted_idx]
    
    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Grafico superiore: Frequenza di predizione Positiva
    colors = ['#1f77b4' if val == 0 else '#d62728' for val in sorted_actual] # Blu: Sano, Rosso: Malato
    ax1.scatter(range(len(sorted_stability)), sorted_stability, 
                c=colors, alpha=0.4, s=20, label='Punti: Blu (H), Rossi (P)')
    
    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    ax1.set_ylabel("Frequenza Predizione '1'")
    ax1.set_title(f"{title}\n(Ogni punto è un campione del dataset valutato su {n_runs} run)")
    ax1.legend()
    ax1.grid(True, alpha=0.15)

    # Grafico inferiore: Indice di Instabilità
    ax2.fill_between(range(len(sorted_stability)), 0, uncertainty[sorted_idx], 
                     color='orange', alpha=0.4, label='Grado di Incertezza (Varianza)')
    ax2.set_ylabel("Instabilità")
    ax2.set_xlabel("Campioni ordinati per confidenza")
    ax2.legend()
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig("Immagini/"+scenario+"/"+save_path, dpi=300)

    plt.show()

    # Metriche sintetiche
    stable_mask = (stability_mean <= 0.1) | (stability_mean >= 0.9)
    print(f"Campioni con predizione stabile (>90% delle run): {np.sum(stable_mask)}/{n_samples_total}")

#============ OTHER SCENARIOS PLOTS ============

def plot_configs_box_plot(results : Dict,scenario ="_",save_path ="_accuracy_boxplot", title="Box Plot Accuracy per Configurazione"):

    configs = []
    test_scores = []

    for config, cv_logger in results.items():

        configs.append(config)
        test_score = cv_logger.test_scores
        test_scores.append(test_score)

    plt.figure(figsize=(12,6))

    plt.boxplot(test_scores)

    plt.xticks(range(1,len(configs)+1), configs, rotation=45, ha='right')

    plt.title(title)
    plt.xlabel('Configurazioni')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout() 
    plt.savefig("Immagini/"+scenario+"/"+save_path, dpi=300)
    plt.show()

