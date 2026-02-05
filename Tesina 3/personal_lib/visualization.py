
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
    """
    Visualizza le curve di apprendimento 
    """


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
    """
    Visualizza l'accuracy per epoca durante l'addestramento
    """
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
    """
    Visualizza la distribuzione dell'accuracy durante la cross-validation
    """

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

    """
    Grafica il box plot dell'accuracy della configurazione
    """

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
    Grafica l'incertezza sulla classificazione dei campioni durante le run

    y_true_list: fold di tutte le run
    y_pred_list: predizioni di tutte le run
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
def plot_configs_box_plot_all(results: dict, scenario="_", save_path="_accuracy_boxplot_all", title="Box Plot Accuracy"):
    print(scenario)
    configs = []
    test_scores = []

    for config, cv_logger in results.items():
        configs.append(config)
        test_scores.append(cv_logger.test_scores)

    plt.figure(figsize=(14, 6)) 

    # --- NUOVA LOGICA GAP DINAMICO ---
    positions = []
    current_pos = 1
    
    # Definiamo ogni quanto deve esserci un gap in base allo scenario
    if scenario == "architecture_with_sel":
        gap_every = 4
    elif scenario == "regulation_with_sel":
        gap_every = 3
    elif scenario == "learning_rate_with_sel":
        gap_every = 9
    else:
        gap_every = None  # Nessun gap

    if gap_every:
        for i in range(len(configs)):
            positions.append(current_pos)
            if (i + 1) % gap_every == 0:
                current_pos += 2  
            else:
                current_pos += 1
    else:
        positions = range(1, len(configs) + 1)
    # ---------------------------------

    plt.boxplot(test_scores, positions=positions)

    # Use the calculated positions for the x-ticks as well
    plt.xticks(positions, configs, rotation=45, ha='right')

    plt.title(title)
    plt.xlabel('Configurazioni')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout() 
    plt.savefig(f"Immagini/{scenario}/{save_path}", dpi=300)
    plt.show()

def plot_configs_exec_time_all(results: dict, scenario="_", save_path="_exec_time_all", title="Tempi di Esecuzione"):
    exec_times = []
    configs = []

    for config, cv_logger in results.items():
        configs.append(config)
        summary = cv_logger.get_summary()
        exec_times.append(summary['total_time'])

    plt.figure(figsize=(12, 6))

    # --- NUOVA LOGICA GAP DINAMICO ---
    positions = []
    current_pos = 1
    
    # Definiamo ogni quanto deve esserci un gap in base allo scenario
    if scenario == "architecture_with_sel":
        gap_every = 4
    elif scenario == "regulation_with_sel":
        gap_every = 3
    elif scenario == "learning_rate_with_sel":
        gap_every = 9
    else:
        gap_every = None  # Nessun gap

    if gap_every:
        for i in range(len(configs)):
            positions.append(current_pos)
            if (i + 1) % gap_every == 0:
                current_pos += 2  
            else:
                current_pos += 1
    else:
        positions = range(1, len(configs) + 1)
    # ---------------------------------

    bars = plt.bar(positions, exec_times, color='mediumseagreen', edgecolor='black')
    
    dynamic_font_size = max(5, min(12, 120 / len(configs)))
    plt.bar_label(bars, padding=3, fmt='%.2f', fontsize=dynamic_font_size, fontweight='bold')

    plt.title(title, fontsize=14)
    plt.xlabel("Configurazioni", fontsize=12)
    plt.ylabel("Tempo di esecuzione (sec)", fontsize=12)

    plt.xticks(positions, configs, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"Immagini/{scenario}/{save_path}", dpi=300)
    plt.show()

def plot_configs_box_plot(results : Dict,scenario ="_",save_path ="_accuracy_boxplot", title="Box Plot Accuracy (Top 10 Configurazioni)"):
    """
    Grafica i box plot dell'accuracy di ogni configurazione
    """
    print(scenario)
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

def plot_configs_exec_time(results : Dict, scenario ="_",save_path ="_exec_time",title ="Tempi di Esecuzione (Top 10 Configurazioni)"):
    """
    Grafica i tempi di esecuzione di ogni configurazione
    """

    exec_times = []
    configs = []

    for config, cv_logger in results.items():

        configs.append(config)
        summary = cv_logger.get_summary()
        exec_time = summary['total_time']

        exec_times.append(exec_time)

    plt.figure(figsize=(10,6))
    dynamic_font_size = max(6, min(12, 120 / len(configs)))
    bars = plt.bar(configs,exec_times, color='mediumseagreen', edgecolor='black')
    plt.bar_label(bars,padding=3, fmt='%.2f', fontsize=dynamic_font_size, fontweight='bold')

    plt.title("Tempo medio di esecuzione per configurazione", fontsize=14)
    plt.xlabel("Configurazioni", fontsize=12)
    plt.ylabel("Tempo di esecuzione (sec)", fontsize=12)

    plt.xticks(rotation=45, ha="right")


    plt.tight_layout()
    plt.savefig("Immagini/"+scenario+"/"+save_path, dpi=300)
    plt.show()

def plot_cv_stability_comparison(results_y_true: Dict, results_y_pred: Dict, n_runs=30, scenario="_", save_path="comparison_stability"):
    """
    Confronta la stabilità di tutte le configurazioni in un unico grafico a barre raggruppate.
    """
    
    def flatten(input_list):
        flat = []
        for item in input_list:
            if isinstance(item, (pd.Series, np.ndarray)): flat.extend(item.tolist())
            elif isinstance(item, list): flat.extend(item)
            else: flat.append(item)
        return np.array(flat)

    # 1. Preparazione dati
    bins = np.linspace(0, 1, 11)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    
    all_counts = {}
    configs = list(results_y_true.keys())

    for config in configs:
        y_pred_all = flatten(results_y_pred[config])
        n_samples_total = len(y_pred_all) // n_runs
        preds_matrix = y_pred_all.reshape(n_runs, n_samples_total)
        stability_mean = np.mean(preds_matrix, axis=0)
        
        counts, _ = np.histogram(stability_mean, bins=bins)
        all_counts[config] = counts

    # 2. Parametri del plot
    x = np.arange(len(bin_labels))  # Posizioni dei bin
    width = 0.8 / len(configs)       # Larghezza dinamica delle barre basata sul numero di config
    
    fig, ax = plt.subplots(figsize=(14, 8))

    # 3. Creazione delle barre raggruppate
    for i, config in enumerate(configs):
        offset = (i - (len(configs) - 1) / 2) * width
        ax.bar(x + offset, all_counts[config], width, label=config, edgecolor='black', alpha=0.8)

    # 4. Formattazione
    ax.set_title("Confronto Distribuzione Stabilità (Top 10 Configurazioni)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Frequenza di predizione Classe '1' (0=Stabile 0, 1=Stabile 1)", fontsize=12)
    ax.set_ylabel("Numero di Campioni", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.legend(title="Configurazioni", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"Immagini/{scenario}/{save_path}.png", dpi=300)
    plt.show()



def plot_config_roc_curves(results: Dict, scenario="_",save_path ="_roc_curves", title="Confronto Curve ROC (Top 10 Configurazioni)"):
    """
    Curve ROC delle TOP 10 configurazioni per mean_test_acc,
    con evidenziazione del punto più vicino a (0,1)
    e colorazione basata sull'accuracy media.
    
    """

    

    accs = np.array([cv.get_summary()['mean_test_acc'] for _, cv in results.items()])
    acc_min, acc_max = accs.min(), accs.max()

    def normalize(acc):
        return (acc - acc_min) / (acc_max - acc_min + 1e-8)

    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(10, 8))


    for i, (config_name, cv_logger) in enumerate(results.items()):
        summary = cv_logger.get_summary()

        y_true_all = np.concatenate(summary['y_true'])
        y_proba_all = np.concatenate(summary['y_proba'])

        y_score = y_proba_all[:, 1] if y_proba_all.ndim == 2 else y_proba_all

        fpr, tpr, _ = roc_curve(y_true_all, y_score)
        roc_auc = auc(fpr, tpr)

        # punto più vicino a (0,1)
        distances = np.sqrt((1 - tpr)**2 + fpr**2)
        best_idx = np.argmin(distances)

        # colore basato su accuracy
        acc = summary['mean_test_acc']
        color = cmap(normalize(acc))

        # curva ROC
        plt.plot(
            fpr, tpr,
            color=color,
            lw=2.5,
            alpha=0.9,
            label=f'{config_name} '
                  f'(Acc={acc:.3f}, AUC={roc_auc:.3f})'
        )

        plt.scatter(
            fpr[best_idx], tpr[best_idx],
            color=color,
            edgecolor='black',
            s=60,
            zorder=5
        )


    plt.plot([0, 1], [0, 1],
             linestyle='--',
             color='gray',
             lw=2,
             alpha=0.6,
             label='Random')


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.4, linestyle='--')

    # Colorbar (accuracy)
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=acc_min, vmax=acc_max)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Mean Test Accuracy', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"Immagini/{scenario}/{save_path}.png", dpi=300)

    plt.show()  


def plot_config_loss_convergence(results_dict: Dict, scenario="_", save_path="_convergence_comparison.png"):
    """
    Visualizza il confronto delle curve di loss media tra diverse configurazioni.
    """
    if not results_dict:
        print("Dizionario dei risultati vuoto.")
        return

    plt.figure(figsize=(12, 7))
    
    # Palette di colori per distinguere le configurazioni
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for (config_name, logger), color in zip(results_dict.items(), colors):
        curves = logger.loss_curves
        if not curves:
            continue

        # 1. Calcolo della curva MEDIA (gestendo lunghezze diverse con NaN)
        max_len = max(len(c) for c in curves)
        curves_matrix = np.full((len(curves), max_len), np.nan)
        for i, curve in enumerate(curves):
            curves_matrix[i, :len(curve)] = curve
        
        mean_curve = np.nanmean(curves_matrix, axis=0)

        # 2. Plot delle singole run (molto trasparenti per non disturbare)
        # Mostriamo le singole curve solo per dare l'idea della varianza
        plt.plot(mean_curve, color=color, linewidth=3, label=f'Media: {config_name}', zorder=3)
        
        # Opzionale: area di deviazione standard invece di mille linee (più pulito)
        std_curve = np.nanstd(curves_matrix, axis=0)
        plt.fill_between(range(max_len), 
                         mean_curve - std_curve, 
                         mean_curve + std_curve, 
                         color=color, alpha=0.1)

    # 3. Formattazione Grafico
    plt.title("Confronto Convergenza Loss (Top 10 Configurazioni)", fontsize=15, fontweight='bold')
    plt.xlabel("Iterazioni (Epoche)", fontsize=12)
    plt.ylabel("Valore Loss", fontsize=12)
    
    # Posizioniamo la legenda fuori se sono molte configurazioni
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"Immagini/{scenario}/{save_path}", dpi=300, bbox_inches='tight')
    plt.show()


def plot_config_epoch_accuracy(epoch_res, scenario="_", save_path="_accuracy_comparison.png"):
    """
    Confronta media e deviazione standard su due grafici affiancati con legende indipendenti.
    L'asse Y della media si adatta automaticamente al valore minimo.
    """
    if not epoch_res:
        print("Dizionario epoch_results vuoto.")
        return

    # Creazione della figura con due subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Utilizzo della colormap tab10
    colors = plt.cm.get_cmap('tab10', len(epoch_res))

    for i, (config_name, epoch_data) in enumerate(epoch_res.items()):
        if epoch_data is None:
            continue

        e = epoch_data["epochs"]
        mv = epoch_data["mean_val_acc"]
        sv = epoch_data["std_val_acc"]
        color = colors(i)

        # --- GRAFICO 1: MEDIA ---
        ax1.plot(e, mv, label=config_name, color=color, linewidth=2.5)
        
        # --- GRAFICO 2: DEVIAZIONE STANDARD ---
        ax2.plot(e, sv, label=config_name, color=color, linewidth=2.5)

    # Configurazione Grafico Media
    ax1.set_title("Media Accuracy di Validazione (Top 10 Configurazioni)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoche", fontsize=12)
    ax1.set_ylabel("Mean Accuracy", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)
    # Autoscale del minimo, tetto fissato a 1.0 (con un piccolo margine)
    ax1.set_ylim(bottom=None, top=1.02) 
    ax1.legend(fontsize=9, loc='best') # Legenda interna al primo grafico

    # Configurazione Grafico Deviazione
    ax2.set_title("Incertezza (Deviazione Standard) (Top 10 Configurazioni)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoche", fontsize=12)
    ax2.set_ylabel("Std Dev", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9, loc='best') # Legenda interna al secondo grafico

    plt.tight_layout()
    plt.savefig(f"Immagini/{scenario}/{save_path}", dpi=300)
    plt.show()


def plot_config_learning_curves(lc_results_dict, scenario="_", save_path="_learning_curves_grid.png"):
    """
    Crea una griglia di Learning Curves, una per ogni configurazione nel dizionario.
    """
    if not lc_results_dict:
        print("Dizionario lc_results vuoto.")
        return

    n_configs = len(lc_results_dict)
    # Calcoliamo righe e colonne per la griglia (es. con 4 config facciamo 2x2)
    cols = 2
    rows = (n_configs + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)
    axes = axes.flatten()

    for i, (config_name, results) in enumerate(lc_results_dict.items()):
        ax = axes[i]
        
        train_sizes = results["train_sizes"]
        mean_train = results["mean_train_score"]
        std_train  = results["std_train_score"]
        mean_val   = results["mean_val_score"]
        std_val    = results["std_val_score"]

        # Plot Train
        ax.plot(train_sizes, mean_train, marker="o", label="Train score", color="#1f77b4")
        ax.fill_between(train_sizes, mean_train - std_train, mean_train + std_train, alpha=0.15, color="#1f77b4")

        # Plot Validation
        ax.plot(train_sizes, mean_val, marker="o", label="Validation score", color="#ff7f0e")
        ax.fill_between(train_sizes, mean_val - std_val, mean_val + std_val, alpha=0.15, color="#ff7f0e")

        ax.set_title(f"Config: {config_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Esempi di training")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(bottom=None, top=1.02) # Autoscale del minimo per vedere meglio i trend
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=9)

    # Rimuoviamo eventuali subplot vuoti se n_configs è dispari
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"Immagini/{scenario}/{save_path}", dpi=300, bbox_inches='tight')
    plt.show()