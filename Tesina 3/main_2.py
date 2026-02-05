import pickle
from typing import Dict, List
import warnings

from personal_lib.visualization import *

from personal_lib.train import *
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")





def main_phase_1(feature_selection : bool = False, filepath : str ="DARWIN.csv", mlp : MLPClassifier = None):
    print("=" * 70)
    print("FASE 1: ANALISI MLP DEFAULT")
    print("=" * 70)

    print("\nCaricamento dataset...")
    X, y = load_dataset(filepath=filepath,selection=feature_selection)
    print(f"   Dataset: {X.shape[0]} istanze, {X.shape[1]} features")
    print(f"   Distribuzione classi: {np.bincount(y)}")


    print("Train/Test split del dataset caricato...")
    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=SEED, stratify=y)


    # ========== CROSS VALIDATION (METODO PRINCIPALE) ==========
    print("\n" + "=" * 70)
    print("CROSS VALIDATION (30 run × 5 fold = 150 valutazioni)")
    print("=" * 70)
    cv_logger, lc_results, epoch_results = train_mlp_with_cv(X, y, n_splits=5, n_runs=30, epoch_mode="all",mlp=mlp)

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

def main_phase_2(mlp_configs : Dict, feature_selection : bool = False, filepath : str ="DARWIN.csv",):

    cv_logger_dict ={}
    lc_results_dict={}
    epoch_results_dict={}

    count=1
    n_config = len(mlp_configs)
    for config, mlp in mlp_configs.items():

       

        print(f"Configurazione {count}/{n_config}") 

        X, y = load_dataset(filepath=filepath,selection=feature_selection)
        cv_logger, lc_results, epoch_results = train_mlp_with_cv(X, y, n_splits=5, n_runs=30, epoch_mode="all",mlp=mlp)

        cv_logger_dict[config]=cv_logger
        lc_results_dict[config]=lc_results
        epoch_results_dict[config]=epoch_results
        count+=1

    return cv_logger_dict, lc_results_dict, epoch_results_dict



if __name__ == "__main__":

    MODE = "plot"              # train, plot
    EXPERIMENT = "architecture"      # default, architecture, learning_rate, regulation
    SELECTION = True           # False (no selezione), True (selezione)

    LOGGER_FILE = "pickles/"+EXPERIMENT+"_cv_logger"
    MODEL_FILE = "pickles/"+EXPERIMENT+"_model"
    LC_FILE = "pickles/"+EXPERIMENT+"_lc"
    EPOCH_FILE = "pickles/"+EXPERIMENT+"_epoch"
    
    WITH_SEL= "_with_sel"
    if SELECTION == False:
        WITH_SEL = "_no_sel"
        
    
    LOGGER_FILE+=WITH_SEL
    MODEL_FILE+=WITH_SEL
    LC_FILE+=WITH_SEL
    EPOCH_FILE+=WITH_SEL

    if MODE == "train":
        if EXPERIMENT == "default":
            cv_logger, split_logger, final_model, lc_results, epoch_results= main_phase_1(SELECTION)
        
        elif EXPERIMENT == "architecture":
            mlp_configs = get_mlp_config(scenario=EXPERIMENT)
            print(f"Generate {len(mlp_configs)} combinazioni di MLP.")
            cv_logger, lc_results, epoch_results = main_phase_2(mlp_configs=mlp_configs, feature_selection=SELECTION)


        elif EXPERIMENT == "learning_rate":
            mlp_configs = get_mlp_config(scenario=EXPERIMENT)
            print(f"Generate {len(mlp_configs)} combinazioni di MLP.")
            cv_logger, lc_results, epoch_results = main_phase_2(mlp_configs=mlp_configs, feature_selection=SELECTION)
            
        
        elif EXPERIMENT == "regulation":
            mlp_configs = get_mlp_config(scenario=EXPERIMENT)
            print(f"Generate {len(mlp_configs)} combinazioni di MLP.")
            cv_logger, lc_results, epoch_results = main_phase_2(mlp_configs=mlp_configs, feature_selection=SELECTION)

        if EXPERIMENT == "default":
            with open(LOGGER_FILE, "wb") as f:
                pickle.dump(cv_logger, f)
            with open(MODEL_FILE, "wb") as f:
                pickle.dump(final_model, f)
            with open(LC_FILE, "wb") as f:
                pickle.dump(lc_results, f)
            with open(EPOCH_FILE, "wb") as f:
                pickle.dump(epoch_results, f)
        else:
            with open(LOGGER_FILE, "wb") as f:
                pickle.dump(cv_logger, f)
            with open(LC_FILE, "wb") as f:
                pickle.dump(lc_results, f)
            with open(EPOCH_FILE, "wb") as f:
                pickle.dump(epoch_results, f)
       
        
    elif MODE == "plot":

        with open(LOGGER_FILE, "rb") as f:
            cv_logger = pickle.load(f)
        with open(LC_FILE, "rb") as f:
            lc_results = pickle.load(f)
        with open(EPOCH_FILE, "rb") as f:
            epoch_results = pickle.load(f)
       
        if EXPERIMENT == "default":
            cv_summary = cv_logger.get_summary()
            y_test = cv_summary['y_true']
            y_pred = cv_summary['y_pred']
            y_proba = cv_summary['y_proba']

            print(f"Tempo totale {cv_summary['total_time']:.2f} sec, {cv_summary['total_time']/60:.2f} min")

            plot_loss_convergence(cv_logger,scenario=EXPERIMENT+WITH_SEL)
            plot_learning_curves(lc_results,scenario=EXPERIMENT+WITH_SEL)
            plot_epoch_accuracy(epoch_results,scenario=EXPERIMENT+WITH_SEL)
            plot_cv_confusion_matrix(y_test, y_pred, ['P','H'],scenario=EXPERIMENT+WITH_SEL)
            plot_cv_roc_curve(y_true_list=y_test,y_proba_list=y_proba,scenario=EXPERIMENT+WITH_SEL)
            plot_accuracy_distribution(cv_logger,scenario=EXPERIMENT+WITH_SEL)
            plot_accuracy_boxplot(cv_logger,scenario=EXPERIMENT+WITH_SEL)
            plot_cv_prediction_stability(y_test,y_pred,scenario=EXPERIMENT+WITH_SEL)

        else:
            
            top_n_config = get_top_n_config(cv_logger)
            y_test={}
            y_pred={}
            y_proba={}

            for config, cv_logger in top_n_config.items():
                cv_summary = cv_logger.get_summary()
                y_test[config]=cv_summary['y_true']
                y_pred[config]= cv_summary['y_pred']
                y_proba[config]= cv_summary['y_proba']

            print(top_n_config)
            plot_configs_box_plot(top_n_config,scenario=EXPERIMENT+WITH_SEL)
            plot_configs_exec_time(top_n_config,scenario=EXPERIMENT+WITH_SEL)
            plot_cv_stability_comparison(y_test,y_pred, scenario=EXPERIMENT+WITH_SEL)
    
