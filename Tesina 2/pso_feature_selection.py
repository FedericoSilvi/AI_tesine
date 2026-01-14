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
import matplotlib.pyplot as plt
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
    # TODO: Implementare il caricamento -DONE

    # Dataset loading
    dataset = pd.read_csv(filepath)

    # Features and class extraction
    features = dataset.iloc[:, 1:-1]
    classes = dataset.iloc[:, -1]

    # Missing values management: median substitution
    features = features.fillna(features.median())

    return (features, classes)


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
        # TODO: Inizializzare la posizione (random se non fornita) -DONE
        # Posizione (binaria)
        if position is not None:
            self.position = position
        else:
            # Genera 1 solo se il random è < 0.1 (10% di probabilità)
            self.position = (np.random.rand(self.n_features) < 0.1).astype(int)

        # Velocità (continua)
        # HINT: Inizializzare in range ragionevole, es. [-4, 4]
        self.velocity = np.random.uniform(-4, 4, size=self.n_features)

        # Personal best
        self.pbest_position = None
        self.pbest_fitness = float('-inf')

        # Fitness corrente
        self.fitness = float('-inf')

    def count_selected_features(self) -> int:
        """Restituisce il numero di features selezionate."""
        # TODO: Implementare -DONE

        return np.sum(self.position)

    def update_pbest(self):
        """Aggiorna il personal best se il fitness corrente è migliore."""
        # TODO: Implementare  -DONE

        if self.pbest_fitness < self.fitness:
            self.pbest_fitness = self.fitness
            self.pbest_position = self.position.copy()


# =============================================================================
# FUNZIONE FITNESS
# =============================================================================
def fitness_correlation_based(particle: Particle, X: pd.DataFrame, y: pd.Series, r_cf: np.ndarray,
                              r_ff: np.ndarray) -> float:
    """
    Calcola il fitness basato sulla correlation analysis.

    HINT:
    - Considerate la correlazione features-classe e features-features
    - Un buon subset ha alta correlazione con la classe e bassa ridondanza
    - Formula suggerita: CFS (Correlation-based Feature Selection)

    Returns:
        float: valore di fitness (più alto = migliore)
    """
    # TODO: Implementare la fitness function -DONE

    # Selected features extraction
    selected = np.where(particle.position == 1)[0]
    k = len(selected)

    # Check on k
    if k == 0:
        return 0.0

    # Mean on the feature-class correlation
    r_cf_mean = np.mean(r_cf[selected])

    # Check if there is only 1 feature
    if k > 1:
        # Submatrix of selected features (dimension k x k)
        sub_matrix = r_ff[np.ix_(selected, selected)]

        # Formula: r_ff_mean = (Total Sum - Diagonal Sum)/(k^2-k)
        r_ff_mean = (np.sum(sub_matrix) - k) / (k ** 2 - k)
    else:
        r_ff_mean = 0.0

    # Fitness computation
    # Formula: CFS = (k*r_cf_mean)/sqrt(k+k*(k-1)*r_ff_mean)

    num = k * r_cf_mean
    den = np.sqrt(k + k * (k - 1) * r_ff_mean)

    if den == 0:
        return 0.0
    else:
        return num / den


# =============================================================================
# AGGIORNAMENTO PSO
# =============================================================================
def v_shaped_prob(x: np.ndarray) -> np.ndarray:
    """
    Funzione V-Shaped per convertire velocità -> probabilità di CAMBIAMENTO.
    Usa la tangente iperbolica assoluta.

    Se v = 0 -> prob = 0 (Nessun cambiamento, la particella è stabile)
    Se v alto -> prob -> 1 (Alta probabilità di invertire il bit)
    """
    # Non serve clipping per tanh, gestisce bene overflow
    return np.abs(np.tanh(x))


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
    # TODO: Implementare -DONE

    r1 = np.random.rand(particle.n_features)
    r2 = np.random.rand(particle.n_features)

    v = w * particle.velocity + c1 * r1 * (particle.pbest_position - particle.position) + c2 * r2 * (
                gbest_position - particle.position)

    v_lim = np.clip(v, -v_max, v_max)

    return v_lim


def update_position(particle: Particle) -> np.ndarray:
    """
    Aggiorna la posizione della particella (Logica V-Shaped).

    Logica:
    - Calcolo probabilità di FLIP (cambio stato) basata sulla velocità
    - Eseguo XOR logico tra posizione vecchia e maschera di cambiamento
    """
    # 1. Calcolo probabilità di cambiare stato
    prob_flip = v_shaped_prob(particle.velocity)

    # 2. Genero numeri casuali per decidere se flippare
    random_values = np.random.rand(particle.n_features)

    # 3. Creo una maschera: True dove devo cambiare valore
    flip_mask = random_values < prob_flip

    # 4. Aggiorno la posizione:
    # Se flip_mask è False (0) -> mantengo valore (0^0=0, 1^0=1)
    # Se flip_mask è True (1)  -> inverto valore (0^1=1, 1^1=0)
    new_position = np.logical_xor(particle.position, flip_mask).astype(int)

    # Gestione caso degenere: se tutte le feature sono 0, ne accendo una random
    if np.sum(new_position) == 0:
        fix_idx = np.random.randint(0, particle.n_features)
        new_position[fix_idx] = 1

    return new_position


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
                 w: float = 0.7,  # inerzia
                 c1: float = 2.0,  # coefficiente cognitivo
                 c2: float = 2.0,  # coefficiente sociale
                 v_max: float = 4.0,  # velocità massima
                 max_iterations: int = 50,
                 convergence_threshold: int = None,  # iterazioni senza miglioramento
                 convergence_tolerance: float = 1e-5,
                 random_seed: int = SEED):

        # TODO: Inizializzare i parametri - DONE
        np.random.seed(random_seed)  # imposto il seed per il generatore randomico

        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.convergence_tolerance = convergence_tolerance

        # Global best
        self.gbest_position = None
        self.gbest_fitness = float('-inf')

    def initialize_swarm(self, n_features: int) -> List[Particle]:
        """Inizializza lo sciame di particelle."""
        # TODO: Implementare - DONE
        return [Particle(n_features) for _ in range(self.swarm_size)]  # Creo lo sciame come una lista di particelle

    def evaluate_swarm(self, swarm: List[Particle],
                       X: pd.DataFrame, y: pd.Series,
                       r_cf: np.ndarray, r_ff: np.ndarray) -> List[float]:
        """
        Valuta il fitness di tutte le particelle dello sciame.

        Per ogni particella viene calcolata la qualità della soluzione
        (sottoinsieme di feature selezionate) tramite una funzione di fitness
        basata sulla correlazione con la variabile target.
        """
        # TODO: Implementare - DONE

        # Lista che conterrà i valori di fitness di tutte le particelle
        fitness_values = []

        # Valuto ogni particella dello sciame
        for particle in swarm:
            # Calcolo della fitness della particella corrente
            particle.fitness = fitness_correlation_based(particle, X, y, r_cf, r_ff)

            # Aggiornamento del best personale (pbest) della particella
            # se la soluzione corrente è migliore di quelle precedenti
            particle.update_pbest()

            # Salvo il valore di fitness per eventuali analisi statistiche
            fitness_values.append(particle.fitness)

        # Restituisco la lista dei valori di fitness dello sciame
        return fitness_values

    def update_gbest(self, swarm: List[Particle]) -> Tuple[np.ndarray, float]:
        """
        Aggiorna il global best (gbest) dello sciame.

        Il global best rappresenta la migliore soluzione trovata
        da tutte le particelle fino all'iterazione corrente.

        Returns:
            Tuple:
                - posizione del global best
                - valore della fitness associata
        """

        # TODO: Implementare - DONE
        # Scorro tutte le particelle dello sciame
        for particle in swarm:

            # Se la fitness della particella corrente è migliore
            # del miglior valore globale trovato finora,
            # aggiorno il global best
            if particle.fitness > self.gbest_fitness:
                self.gbest_fitness = particle.fitness
                self.gbest_position = particle.position.copy()

        # Restituisco posizione e fitness del global best
        return self.gbest_position, self.gbest_fitness

    def run(self, X: pd.DataFrame, y: pd.Series, r_cf: np.ndarray, r_ff: np.ndarray, id_run: int = 1, ) -> Dict:
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
        # TODO: Implementare il loop principale del PSO - DONE
        start_time = time.time()

        logger = ExperimentLogger()

        swarm = self.initialize_swarm(X.shape[1])
        no_improvement = 0

        iterations_completed = 0

        w_start = 0.7
        w_end = 0.7 # lascio fisso
        self.c1 = 1
        self.c2 = 1

        for iteration in range(self.max_iterations):
            #print(iterations_completed)
            iterations_completed += 1

            self.w = w_start - (w_start - w_end) * (iteration / self.max_iterations)

            # 1. Valuto lo stormo ed aggiorno il global best se serve
            self.evaluate_swarm(swarm, X, y, r_cf, r_ff)

            prev_best = self.gbest_fitness
            self.update_gbest(swarm)

            # Metriche per il logging
            avg_fitness = float(np.mean([particle.fitness for particle in swarm]))
            std_fitness = float(np.std([particle.fitness for particle in swarm]))
            dispersion = self.calculate_swarm_dispersion(swarm)
            avg_velocity = self.calculate_average_velocity(swarm)

            # Aggiorno il log dell'iterazione
            logger.log_iteration(
                iteration=iteration,
                gbest_fitness=self.gbest_fitness,
                avg_fitness=avg_fitness,
                std_fitness=std_fitness,
                dispersion=dispersion,
                avg_velocity=avg_velocity,
                selected_features=self.gbest_position
            )

            # 2. Valuto se terminare prematuramente l'algoritmo

            # Controllo se il miglioramento della fitness globale
            # rispetto all'iterazione precedente è inferiore alla soglia di tolleranza
            if abs(self.gbest_fitness - prev_best) < self.convergence_tolerance:
                # Se il miglioramento è trascurabile, incremento il contatore
                # delle iterazioni senza miglioramento significativo
                no_improvement += 1
            else:
                # Se c'è stato un miglioramento apprezzabile,
                # azzero il contatore
                no_improvement = 0

            # Se è stato definito un numero massimo di iterazioni senza miglioramento
            # e tale soglia viene superata, interrompo l'esecuzione dell'algoritmo
            if self.convergence_threshold and no_improvement >= self.convergence_threshold:
                break

            # 3. Aggiorno posizione e velocità delle particelle
            for particle in swarm:
                particle.velocity = update_velocity(particle, self.gbest_position,
                                                    self.w, self.c1, self.c2, self.v_max)
                particle.position = update_position(particle)

        # Aggiorno il log globale
        best_particle = Particle(n_features=len(self.gbest_position))
        best_particle.position = self.gbest_position.copy()
        best_particle.fitness = self.gbest_fitness

        logger.log_run(
            run_id=id_run,  # placeholder, va aggiunto run_id tra i parametri della funzione?
            best_particle=best_particle,
            execution_time=time.time() - start_time,
            iterations_completed=iterations_completed
        )

        return {
            "best_fitness": self.gbest_fitness,
            "best_position": self.gbest_position,
            "selected_features": int(self.gbest_position.sum()),
            "iterations": iterations_completed,
            "execution_time": time.time() - start_time,
            "logger": logger
        }

    def calculate_swarm_dispersion(self, swarm: List[Particle]) -> float:
        """
        Calcola la dispersione dello sciame.

        La dispersione misura quanto le particelle sono distribuite
        nello spazio delle soluzioni ed è calcolata come la distanza
        media delle particelle dal centroide dello sciame.
        """

        # TODO: Implementare - DONE

        positions = np.array([p.position for p in swarm])
        n_particles = len(swarm)
        n_features = positions.shape[1]
        total_dist = 0
        count = 0
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                total_dist += np.sum(positions[i] != positions[j])
                count += 1
        return total_dist / count

    def calculate_average_velocity(self, swarm: List[Particle]) -> float:
        """
        Calcola la velocità media dello sciame.

        HINT: Media delle norme dei vettori velocità
        """
        # TODO: Implementare - DONE
        # Calcolo la media delle velocità delle particelle
        velocities = [np.linalg.norm(particle.velocity) for particle in swarm]
        return float(np.mean(velocities))


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
                      avg_fitness: float, std_fitness: float, dispersion: float,
                      avg_velocity: float, selected_features: np.ndarray):
        """Logga i dati di un'iterazione."""
        # TODO: Implementare - DONE
        iter_data = {
            "iteration": iteration,
            "gbest_fitness": gbest_fitness,
            "avg_fitness": avg_fitness,
            "std_fitness": std_fitness,
            "dispersion": dispersion,
            "avg_velocity": avg_velocity,
            "selected_features_count": int(selected_features.sum())
        }
        self.iterations_data.append(iter_data)

        # Aggiorno la frequenza di selezione di ogni feature
        for idx, sel in enumerate(selected_features):
            if sel:
                self.feature_counts[idx] = self.feature_counts.get(idx, 0) + 1

    def log_run(self, run_id: int, best_particle: Particle,
                execution_time: float, iterations_completed: int):
        """Logga i dati di un run completo."""
        # TODO: Implementare - DONE
        run_data = {
            "run_id": run_id,
            "best_fitness": best_particle.fitness,
            "selected_features_count": best_particle.count_selected_features(),
            "execution_time": execution_time,
            "iterations_completed": iterations_completed,
            "best_position": best_particle.position.copy()
        }
        self.run_times.append(run_data)


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

    logger = results["logger"]

    
    # Estrazione delle grandezze di interesse:
    # - iterazioni 
    # - fitness media 
    # - deviazione standard della fitness
    # - migliori fitness 

    iters = np.array([it["iteration"] for it in logger.iterations_data])
    avg_fitness = np.array([it["avg_fitness"] for it in logger.iterations_data])
    std_fitness = np.array([it["std_fitness"] for it in logger.iterations_data]) # Deviazione standard

    best_fitness = np.array([it["gbest_fitness"] for it in logger.iterations_data])

    opt = results["best_fitness"]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plotting
    ax.plot(iters, avg_fitness, color='orange', linewidth=2, label='Swarm Average (Mean)')
    ax.fill_between(iters, 
                    avg_fitness - std_fitness, 
                    avg_fitness + std_fitness, 
                    color='orange', alpha=0.2, label='± Std Dev')

    # Plot del Best
    ax.plot(iters, best_fitness, color='blue', linewidth=2.5, label='Global Best')

    # Linea ottima
    ax.axhline(opt, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target Optimum')

    # Formattazione
    ax.set_xlim(0, max(iters))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.show()


def plot_fitness_boxplots(results: List[Dict], title: str = "Fitness Distribution"):
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
    # 1. Caricamento dati
    X, y = load_darwin_dataset("Tesina 2/DARWIN.csv")

    # 2. Conversione target da stringa a numerico (P=1, H=0)
    y_numeric = y.map({'P': 1, 'H': 0})

    # 3. Calcolo correlazioni
    r_cf = np.array([np.abs(X.iloc[:, i].corr(y_numeric)) for i in range(X.shape[1])])
    r_ff = np.abs(X.corr().values)

    # 4. Creazione oggetto PSO con parametri di default
    pso = ParticleSwarmOptimization(max_iterations=100)

    # 5. Esecuzione PSO
    result = pso.run(X, y_numeric, r_cf, r_ff, id_run=1)

    # 6. Stampa log iterazioni
    print("\n=== LOG ITERAZIONI ===")
    for iter_data in result['logger'].iterations_data:
        print(f"Iter {iter_data['iteration']:03d} | "
              f"GBest: {iter_data['gbest_fitness']:.4f} | "
              f"Avg: {iter_data['avg_fitness']:.4f} | "
              f"Dispersion: {iter_data['dispersion']:.4f} | "
              f"AvgVel: {iter_data['avg_velocity']:.4f} | "
              f"Selected Features: {iter_data['selected_features_count']}")

    # 7. Stampa riepilogo run
    best_run = result['logger'].run_times[0]  # singolo run
    print("\n=== RUN SUMMARY ===")
    print(f"Best Fitness: {best_run['best_fitness']:.4f}")
    print(f"Selected Features: {best_run['selected_features_count']}")
    print(f"Iterations Completed: {best_run['iterations_completed']}")
    print(f"Execution Time: {best_run['execution_time']:.2f} sec")
    print(f"GBest Position: {best_run['best_position']}")

    print("\n\n\n\n=== FREQUENZA SELEZIONE FEATURES (ORDINATE PER FREQUENZA) ===\n")
    for feature, count in sorted(result['logger'].feature_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"Feature {feature} | Selezionata {count} volte")

    print("\n\n\n\n=== FREQUENZA SELEZIONE FEATURES (ORDINATE PER ID) ===\n")
    for feature, count in sorted(result['logger'].feature_counts.items()):
        print(f"Feature {feature} | Selezionata {count} volte")


    plot_convergence_curves(results=result)