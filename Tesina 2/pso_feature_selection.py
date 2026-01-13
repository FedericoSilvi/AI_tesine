"""
Particle Swarm Optimization for Feature Selection on DARWIN Dataset
====================================================================

Tesina 2 - Analisi PSO come Feature Selection

HINT: Questo script fornisce una struttura base. Dovrete:
- Completare le funzioni con la logica appropriata
- Implementare l'aggiornamento velocità/posizione per problemi binari
- Gestire correttamente i parametri e le metriche
"""

import numpy as np
import pandas as pd
import time
from typing import Tuple, List, Dict

# HINT: Considerate quali altre librerie potrebbero essere utili per 
# la valutazione delle correlazioni e la visualizzazione

# =============================================================================
# CONFIGURAZIONE E SEED
# =============================================================================
SEED = 42  # HINT: Usare lo stesso seed per garantire riproducibilità


# =============================================================================
# CARICAMENTO DATASET
# =============================================================================
def load_darwin_dataset(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carica il dataset DARWIN e gestisce eventuali missing values.
    
    HINT: 
    - La prima colonna è l'ID, l'ultima è la classe (P/H)
    - Le colonne intermedie sono le 450 features (25 task × 18 features)
    - Considerate diverse strategie per i missing values
    
    Returns:
        X: DataFrame delle features
        y: Series delle classi
    """
    # TODO: Implementare il caricamento

    # Dataset loading 
    dataset = pd.read_csv(filepath)

    # Features and class extraction
    features = dataset.iloc[:,1:-1] 
    classes = dataset.iloc[:,-1]

    # Missing values management: median substitution
    features = features.fillna(features.median())
    
    return (features,classes)


# =============================================================================
# RAPPRESENTAZIONE PARTICELLA
# =============================================================================
class Particle:
    """
    Rappresenta una particella nello sciame.
    
    HINT: 
    - Posizione binaria: 1 = feature selezionata, 0 = feature non selezionata
    - Velocità: valori continui che influenzano la probabilità di selezione
    - Considerate la funzione sigmoid per convertire velocità in probabilità
    """
    
    def __init__(self, n_features: int, position: np.ndarray = None):
        self.n_features = n_features
        
        # Posizione (binaria)
        # TODO: Inizializzare la posizione (random se non fornita)
        if position is not None:
            self.position=position
        else:
         self.position = np.random.randint(0,2,size=self.n_features)
        
        # Velocità (continua)
        # HINT: Inizializzare in range ragionevole, es. [-4, 4]
        self.velocity = np.random.uniform(-4,4,size=self.n_features)
        
        # Personal best
        self.pbest_position = None
        self.pbest_fitness = float('-inf')
        
        # Fitness corrente
        self.fitness = float('-inf')
    
    def count_selected_features(self) -> int:
        """Restituisce il numero di features selezionate."""
        # TODO: Implementare

        return np.sum(self.position)

    
    def update_pbest(self):
        """Aggiorna il personal best se il fitness corrente è migliore."""
        # TODO: Implementare

        if self.pbest_fitness<self.fitness:
            self.pbest_fitness = self.fitness


# =============================================================================
# FUNZIONE FITNESS
# =============================================================================
def fitness_correlation_based(particle: Particle, X: pd.DataFrame, y: pd.Series, r_cf: np.ndarray, r_ff: np.ndarray) -> float:
    """
    Calcola il fitness basato sulla correlation analysis.
    
    HINT:
    - Considerate la correlazione features-classe e features-features
    - Un buon subset ha alta correlazione con la classe e bassa ridondanza
    - Formula suggerita: CFS (Correlation-based Feature Selection)
    
    Returns:
        float: valore di fitness (più alto = migliore)
    """
    # TODO: Implementare la fitness function

    # Selected features extraction 
    selected = np.where(particle.position==1)[0]
    k = len(selected)

    # Check on k
    if k==0:
        return 0.0

    # Mean on the feature-feature correlation 
    r_cf_mean = np.mean(r_cf[selected])

    # Check if there is only 1 feature
    if k>1:
        # Submatrix of selected features (dimension k x k)
        sub_matrix = r_ff[np.ix_(selected,selected)] 

        # Formula: r_ff_mean = (Total Sum - Diagonal Sum)/(k^2-k)
        r_ff_mean = (np.sum(sub_matrix)-k)/(k**2-k)
    else:
        r_ff_mean= 0.0

    # Fitness computation
    # Formula: CFS = (k*r_cf_mean)/sqrt(k+k*(k-1)*r_ff_mean)

    num = k*r_cf_mean
    den = np.sqrt(k+k*(k-1)*r_ff_mean)

    if den == 0:
        return 0.0
    else:
        return num/den

# =============================================================================
# AGGIORNAMENTO PSO
# =============================================================================
def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Funzione sigmoid per conversione velocità -> probabilità.
    
    HINT: Gestire overflow per valori molto grandi/piccoli di x
    """
    # TODO: Implementare con gestione overflow

    # Clipping the velocity in order to avoid overflow
    x_check = np.clip(x,-50,50)
    # Sigmoid computation 
    z = 1/(1+np.exp(-x_check))

    return z


def update_velocity(particle: Particle, gbest_position: np.ndarray,
                    w: float = 0.7, c1: float = 2.0, c2: float = 2.0,
                    v_max: float = 4.0) -> np.ndarray:
    """
    Aggiorna la velocità della particella.
    
    HINT: 
    - Formula: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    - Limitare la velocità in [-v_max, v_max]
    - r1, r2 sono vettori random in [0,1]
    
    Returns:
        np.ndarray: nuova velocità
    """
    # TODO: Implementare
    
    r1 = np.random.rand(particle.n_features)
    r2 = np.random.rand(particle.n_features)

    v = w*v + c1*r1*(particle.pbest_position - particle.position) + c2*r2*(gbest_position - particle.position)

    v_lim = np.clip(v,-v_max,v_max)

    return v_lim 


def update_position(particle: Particle) -> np.ndarray:
    """
    Aggiorna la posizione della particella (versione binaria).
    
    HINT:
    - Usare sigmoid(velocity) come probabilità
    - La nuova posizione è 1 se random < sigmoid(v), altrimenti 0
    - Gestire il caso in cui nessuna feature è selezionata
    
    Returns:
        np.ndarray: nuova posizione binaria
    """
    # TODO: Implementare
    pass


# =============================================================================
# ALGORITMO PSO
# =============================================================================
class ParticleSwarmOptimization:
    """
    Implementazione dell'Algoritmo PSO per Feature Selection.
    
    HINT: Questa classe dovrebbe essere modulare per permettere
    l'analisi parametrica richiesta dalla tesina.
    """
    
    def __init__(self, 
                 swarm_size: int = 100,
                 w: float = 0.7,           # inerzia
                 c1: float = 2.0,          # coefficiente cognitivo
                 c2: float = 2.0,          # coefficiente sociale
                 v_max: float = 4.0,       # velocità massima
                 max_iterations: int = 100,
                 convergence_threshold: int = None,  # iterazioni senza miglioramento
                 convergence_tolerance: float = 1e-5,
                 random_seed: int = SEED):
        
        # TODO: Inizializzare i parametri
        pass
    
    def initialize_swarm(self, n_features: int) -> List[Particle]:
        """Inizializza lo sciame di particelle."""
        # TODO: Implementare
        pass
    
    def evaluate_swarm(self, swarm: List[Particle], 
                       X: pd.DataFrame, y: pd.Series) -> List[float]:
        """Valuta il fitness di tutte le particelle."""
        # TODO: Implementare
        pass
    
    def update_gbest(self, swarm: List[Particle]) -> Tuple[np.ndarray, float]:
        """
        Aggiorna il global best.
        
        Returns:
            Tuple: (posizione gbest, fitness gbest)
        """
        # TODO: Implementare
        pass
    
    def run(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Esegue l'algoritmo PSO.
        
        HINT: Implementare logging dettagliato di:
        - Best/average fitness per iterazione
        - Features selezionate nella migliore soluzione
        - Tempo di esecuzione
        - Metriche dello sciame (dispersione, velocità media)
        
        Returns:
            Dict con risultati e metriche
        """
        # TODO: Implementare il loop principale del PSO
        pass
    
    def calculate_swarm_dispersion(self, swarm: List[Particle]) -> float:
        """
        Calcola la dispersione dello sciame.
        
        HINT: Distanza media delle particelle dal centroide
        """
        # TODO: Implementare
        pass
    
    def calculate_average_velocity(self, swarm: List[Particle]) -> float:
        """
        Calcola la velocità media dello sciame.
        
        HINT: Media delle norme dei vettori velocità
        """
        # TODO: Implementare
        pass


# =============================================================================
# LOGGING E METRICHE
# =============================================================================
class ExperimentLogger:
    """
    Logger per gli esperimenti.
    
    HINT: Salvare tutte le informazioni necessarie per generare:
    - Curve di convergenza
    - Box plot delle distribuzioni
    - Frequenza di selezione delle features
    - Analisi comportamento sciame
    """
    
    def __init__(self):
        self.iterations_data = []
        self.run_times = []
        self.feature_counts = {}  # frequenza selezione per feature
        self.swarm_behavior = []  # dispersione, velocità media, etc.
    
    def log_iteration(self, iteration: int, gbest_fitness: float, 
                      avg_fitness: float, dispersion: float,
                      avg_velocity: float, selected_features: np.ndarray):
        """Logga i dati di un'iterazione."""
        # TODO: Implementare
        pass
    
    def log_run(self, run_id: int, best_particle: Particle, 
                execution_time: float, iterations_completed: int):
        """Logga i dati di un run completo."""
        # TODO: Implementare
        pass


# =============================================================================
# ESPERIMENTI PARAMETRICI
# =============================================================================

def run_experiment_swarm_size(X: pd.DataFrame, y: pd.Series, 
                              n_runs: int = 30) -> Dict:
    """
    Scenario 1: Test dimensioni sciame [20, 50, 100, 200, 500]
    
    HINT: Parametri fissi: w=0.7, c1=2.0, c2=2.0
    
    Returns:
        Dict con risultati aggregati per ogni dimensione
    """
    swarm_sizes = [20, 50, 100, 200, 500]
    # TODO: Implementare ciclo esperimenti
    pass


def run_experiment_pso_coefficients(X: pd.DataFrame, y: pd.Series,
                                    n_runs: int = 30) -> Dict:
    """
    Scenario 2: Test coefficienti PSO
    
    HINT:
    - Inerzia (w): [0.4, 0.6, 0.7, 0.9]
    - Coefficiente cognitivo (c1): [1.0, 1.5, 2.0, 2.5]
    - Coefficiente sociale (c2): [1.0, 1.5, 2.0, 2.5]
    - Dimensione sciame fissa: 100
    
    Analisi: bilanciamento esplorazione/sfruttamento
    
    Returns:
        Dict con risultati per ogni combinazione
    """
    inertia_values = [0.4, 0.6, 0.7, 0.9]
    c1_values = [1.0, 1.5, 2.0, 2.5]
    c2_values = [1.0, 1.5, 2.0, 2.5]
    # TODO: Implementare griglia di esperimenti
    pass


def run_experiment_stopping_criteria(X: pd.DataFrame, y: pd.Series,
                                     n_runs: int = 30) -> Dict:
    """
    Scenario 3: Test criteri di stop
    
    HINT:
    - Iterazioni fisse: [50, 100, 200]
    - Convergenza (soglie): [10, 20, 30] iterazioni senza miglioramento
    - Tolleranze: [1e-4, 1e-5, 1e-6]
    
    Returns:
        Dict con risultati per ogni criterio
    """
    fixed_iterations = [50, 100, 200]
    convergence_thresholds = [10, 20, 30]
    tolerances = [1e-4, 1e-5, 1e-6]
    # TODO: Implementare esperimenti
    pass


# =============================================================================
# VISUALIZZAZIONE
# =============================================================================
def plot_convergence_curves(results: Dict, title: str = "Convergence Curves"):
    """
    Genera curve di convergenza.
    
    HINT: Media ± deviazione standard su tutti i run
    """
    # TODO: Implementare con matplotlib
    pass


def plot_fitness_boxplots(results: Dict, title: str = "Fitness Distribution"):
    """
    Genera box plot delle distribuzioni fitness.
    
    HINT: Un boxplot per ogni configurazione testata
    """
    # TODO: Implementare
    pass


def plot_feature_frequency(feature_counts: Dict, feature_names: List[str]):
    """
    Istogramma frequenza selezione features.
    
    HINT: Mostrare le top-k features più frequentemente selezionate
    """
    # TODO: Implementare
    pass


def plot_swarm_behavior(swarm_data: List[Dict], title: str = "Swarm Behavior"):
    """
    Visualizza il comportamento dello sciame nel tempo.
    
    HINT: 
    - Dispersione delle particelle
    - Velocità media
    - Evoluzione del gbest
    """
    # TODO: Implementare
    pass


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # HINT: Struttura suggerita per eseguire tutti gli esperimenti
    
    # 1. Caricamento dati
    # X, y = load_darwin_dataset("DARWIN.csv")
    
    # 2. Esecuzione scenari
    # results_swarm = run_experiment_swarm_size(X, y)
    # results_coefficients = run_experiment_pso_coefficients(X, y)
    # results_stopping = run_experiment_stopping_criteria(X, y)
    
    # 3. Generazione output
    # - Salvare risultati
    # - Generare grafici
    # - Report statistico
    # - Analisi comportamento sciame
    
    print("Eseguire gli esperimenti rimuovendo i commenti sopra.")
    print("Ricordate: minimo 30 run per configurazione!")
