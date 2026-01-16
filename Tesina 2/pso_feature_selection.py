"""
Particle Swarm Optimization for Feature Selection on DARWIN Dataset
====================================================================

Tesina 2 - Analisi PSO come Feature Selection

HINT: Questo script fornisce una struttura base. Dovrete:
- Completare le funzioni con la logica appropriata
- Implementare l'aggiornamento velocità/posizione per problemi binari
- Gestire correttamente i parametri e le metriche
"""
import sys

import numpy as np
import pandas as pd
import time
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from itertools import product

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
def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Funzione sigmoid per conversione velocità -> probabilità.

    HINT: Gestire overflow per valori molto grandi/piccoli di x
    """
    # TODO: Implementare con gestione overflow

    # Clipping the velocity in order to avoid overflow
    x_check = np.clip(x, -4, 4)
    # Sigmoid computation 
    z = 1 / (1 + np.exp(-x_check))

    return z


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
                 max_iterations: int = 100,
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

        """ w_start = 0.7
        w_end = 0.7  # lascio fisso
        self.c1 = 1
        self.c2 = 1 """

        for iteration in range(self.max_iterations):
            # print(iterations_completed)
            iterations_completed += 1

            #self.w = w_start - (w_start - w_end) * (iteration / self.max_iterations)

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

    # TODO: Implementare ciclo esperimenti

    swarm_sizes = [20, 50, 100, 200, 500]
    results = {}

    # File di output
    output_file = "experiment_swarm_size.txt"

    # Calcolo correlazioni
    y_numeric = y
    r_cf = np.array([np.abs(X.iloc[:, i].corr(y_numeric)) for i in range(X.shape[1])])
    r_ff = np.abs(X.corr().values)

    global_run_id = 1

    # Creazione file di testo con tutti i risultati 
    with open(output_file, "w", encoding="utf-8") as f:

        f.write("=== ESPERIMENTO: SWARM SIZE ===\n\n")

        # Scorro per ogni dimensione dello sciame 
        for swarm_size in swarm_sizes:

            print(f"\n=== Avvio test swarm size = {swarm_size} ===")
            f.write(f"\n\n==============================\n")
            f.write(f"SWARM SIZE = {swarm_size}\n")
            f.write(f"==============================\n\n")

            results[swarm_size] = []

            # Ripeto per il numero di run specificato 
            for run_idx in range(1, n_runs + 1):

                print(f"[Swarm size = {swarm_size}] Run {run_idx}/{n_runs} in corso...")

                pso = ParticleSwarmOptimization(
                    swarm_size=swarm_size,
                    max_iterations=100
                )

                result = pso.run(
                    X=X,
                    y=y_numeric,
                    r_cf=r_cf,
                    r_ff=r_ff,
                    id_run=global_run_id
                )

                logger = result["logger"]

                # Salvataggio su file
                f.write(f"--- RUN ID {global_run_id} ---\n")
                f.write(f"Swarm size: {swarm_size}\n")
                f.write(f"Best fitness: {result['best_fitness']:.6f}\n")
                f.write(f"Selected features: {result['selected_features']}\n")
                f.write(f"Iterations: {result['iterations']}\n")
                f.write(f"Execution time: {result['execution_time']:.2f} sec\n\n")

                f.write("LOG ITERAZIONI:\n")
                for it in logger.iterations_data:
                    f.write(
                        f"Iter {it['iteration']:03d} | "
                        f"GBest {it['gbest_fitness']:.6f} | "
                        f"Avg {it['avg_fitness']:.6f} | "
                        f"Std {it['std_fitness']:.6f} | "
                        f"Disp {it['dispersion']:.6f} | "
                        f"AvgVel {it['avg_velocity']:.6f} | "
                        f"Selected {it['selected_features_count']}\n"
                    )

                f.write("\nFREQUENZA SELEZIONE FEATURES:\n")
                for feat, cnt in sorted(logger.feature_counts.items()):
                    f.write(f"Feature {feat} -> {cnt}\n")

                f.write("\n\n")

                # Salvataggio risultati
                results[swarm_size].append(result)

                print(f"[Swarm size = {swarm_size}] Run {run_idx}/{n_runs} COMPLETATA")

                global_run_id += 1

    print("\n=== ESPERIMENTO SWARM SIZE COMPLETATO ===")
    print(f"Log salvati in: {output_file}")

    return results


def run_experiment_pso_coefficients(X: pd.DataFrame, y: pd.Series,
                                    n_runs: int = 30,
                                    section_id: int = 1,
                                    n_sections: int = 1) -> Dict:
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


    all_combinations = list(product(inertia_values, c1_values, c2_values))

    # Suddivisione in sezioni
    # Ogni sezione prende le combinazioni i % n_sections == section_id-1
    section_combinations = [comb for i, comb in enumerate(all_combinations) if i % n_sections == section_id - 1]

    print("=== TUTTE LE COMBINAZIONI ===")
    for i, comb in enumerate(all_combinations, 1):
        print(f"{i:02d}: Inertia={comb[0]}, Cognitive coefficient={comb[1]}, Social coefficient={comb[2]}")

    print(f"\n=== COMBINAZIONI ASSEGNATE ALLA SEZIONE {section_id}/{n_sections} ===")
    for i, comb in enumerate(section_combinations, 1):
        print(f"{i:02d}: Inertia={comb[0]}, Cognitive coefficient={comb[1]}, Social coefficient={comb[2]}")

    results = {}
    output_file = f"experiment_coefficients_section_{section_id}.txt"

    # Correlazioni per fitness
    y_numeric = y
    r_cf = np.array([np.abs(X.iloc[:, i].corr(y_numeric)) for i in range(X.shape[1])])
    r_ff = np.abs(X.corr().values)

    global_run_id = 1

    # Creazione file di testo con tutti i risultati
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== ESPERIMENTO STOPPING CRITERIA - SEZIONE {section_id}/{n_sections} ===\n\n")

        # Scorro sulle sezioni 
        for inertia, c1, c2 in section_combinations:

            key = (inertia, c1, c2)
            results[key] = []

            f.write("\n\n==============================\n")
            f.write(f"Inertia: {inertia} | Cognitive coefficient: {c1} | Social coefficient: {c2}\n")
            f.write("==============================\n\n")

            print(f"\n=== Avvio test Inertia={inertia}, Cognitive coefficient={c1}, Social coefficient={c2} ===")

            for run_idx in range(1, n_runs + 1):
                print(f"[Run {run_idx}/{n_runs}] in corso...")
                pso = ParticleSwarmOptimization(
                    swarm_size=100,
                    w=inertia,
                    c1=c1,
                    c2=c2
                )

                result = pso.run(
                    X=X,
                    y=y_numeric,
                    r_cf=r_cf,
                    r_ff=r_ff,
                    id_run=global_run_id
                )

                logger = result["logger"]

                # Logging su file
                f.write(f"--- RUN ID {global_run_id} ---\n")
                f.write(f"Inertia: {inertia}\n")
                f.write(f"Cognitive coefficient: {c1}\n")
                f.write(f"Social coefficient: {c2}\n")
                f.write(f"Best fitness: {result['best_fitness']:.6f}\n")
                f.write(f"Selected features: {result['selected_features']}\n")
                f.write(f"Iterations: {result['iterations']}\n")
                f.write(f"Execution time: {result['execution_time']:.2f} sec\n\n")

                f.write("LOG ITERAZIONI:\n")
                for it in logger.iterations_data:
                    f.write(
                        f"Iter {it['iteration']:03d} | "
                        f"GBest {it['gbest_fitness']:.6f} | "
                        f"Avg {it['avg_fitness']:.6f} | "
                        f"Std {it['std_fitness']:.6f} | "
                        f"Disp {it['dispersion']:.6f} | "
                        f"AvgVel {it['avg_velocity']:.6f} | "
                        f"Selected {it['selected_features_count']}\n"
                    )

                f.write("\nFREQUENZA SELEZIONE FEATURES:\n")
                for feat, cnt in sorted(logger.feature_counts.items()):
                    f.write(f"Feature {feat} -> {cnt}\n")

                f.write("\n\n")

                results[key].append(result)
                global_run_id += 1

    print(f"\n=== SEZIONE {section_id}/{n_sections} COMPLETATA ===")
    print(f"Log salvati in: {output_file}")

    return results 

def run_experiment_stopping_criteria(X: pd.DataFrame, y: pd.Series,
                                     n_runs: int = 30,
                                     section_id: int = 1,
                                     n_sections: int = 1) -> dict:
    """
    Scenario 3: Test criteri di stop, con suddivisione in sezioni per più computer.

    Args:
        X, y: dataset
        n_runs: numero di run per combinazione
        section_id: sezione da eseguire (1..n_sections)
        n_sections: in quante sezioni dividere il test

    Returns:
        dict dei risultati della sezione
    """

    fixed_iterations = [50, 100, 200]
    convergence_thresholds = [10, 20, 30]
    tolerances = [1e-4, 1e-5, 1e-6]

    all_combinations = list(product(fixed_iterations, convergence_thresholds, tolerances))
    total_combinations = len(all_combinations)

    # Suddivisione in sezioni
    # Ogni sezione prende le combinazioni i % n_sections == section_id-1
    section_combinations = [comb for i, comb in enumerate(all_combinations) if i % n_sections == section_id - 1]

    print("=== TUTTE LE COMBINAZIONI ===")
    for i, comb in enumerate(all_combinations, 1):
        print(f"{i:02d}: MaxIter={comb[0]}, Threshold={comb[1]}, Tol={comb[2]}")

    print(f"\n=== COMBINAZIONI ASSEGNATE ALLA SEZIONE {section_id}/{n_sections} ===")
    for i, comb in enumerate(section_combinations, 1):
        print(f"{i:02d}: MaxIter={comb[0]}, Threshold={comb[1]}, Tol={comb[2]}")

    results = {}
    output_file = f"experiment_stopping_criteria_section_{section_id}.txt"

    # Correlazioni per fitness
    y_numeric = y
    r_cf = np.array([np.abs(X.iloc[:, i].corr(y_numeric)) for i in range(X.shape[1])])
    r_ff = np.abs(X.corr().values)

    global_run_id = 1

    # Creazione file di testo con tutti i risultati
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== ESPERIMENTO STOPPING CRITERIA - SEZIONE {section_id}/{n_sections} ===\n\n")

        # Scorro sulle sezioni 
        for max_iter, threshold, tol in section_combinations:

            key = (max_iter, threshold, tol)
            results[key] = []

            f.write("\n\n==============================\n")
            f.write(f"Max Iter: {max_iter} | Conv Thresh: {threshold} | Tol: {tol}\n")
            f.write("==============================\n\n")

            print(f"\n=== Avvio test Max Iter={max_iter}, Conv Thresh={threshold}, Tol={tol} ===")

            for run_idx in range(1, n_runs + 1):
                print(f"[Run {run_idx}/{n_runs}] in corso...")

                pso = ParticleSwarmOptimization(
                    swarm_size=100,
                    max_iterations=max_iter,
                    convergence_threshold=threshold,
                    convergence_tolerance=tol
                )

                result = pso.run(
                    X=X,
                    y=y_numeric,
                    r_cf=r_cf,
                    r_ff=r_ff,
                    id_run=global_run_id
                )

                logger = result["logger"]

                # Logging su file
                f.write(f"--- RUN ID {global_run_id} ---\n")
                f.write(f"Max Iter: {max_iter}\n")
                f.write(f"Conv Thresh: {threshold}\n")
                f.write(f"Tolerance: {tol}\n")
                f.write(f"Best fitness: {result['best_fitness']:.6f}\n")
                f.write(f"Selected features: {result['selected_features']}\n")
                f.write(f"Iterations: {result['iterations']}\n")
                f.write(f"Execution time: {result['execution_time']:.2f} sec\n\n")

                f.write("LOG ITERAZIONI:\n")
                for it in logger.iterations_data:
                    f.write(
                        f"Iter {it['iteration']:03d} | "
                        f"GBest {it['gbest_fitness']:.6f} | "
                        f"Avg {it['avg_fitness']:.6f} | "
                        f"Std {it['std_fitness']:.6f} | "
                        f"Disp {it['dispersion']:.6f} | "
                        f"AvgVel {it['avg_velocity']:.6f} | "
                        f"Selected {it['selected_features_count']}\n"
                    )

                f.write("\nFREQUENZA SELEZIONE FEATURES:\n")
                for feat, cnt in sorted(logger.feature_counts.items()):
                    f.write(f"Feature {feat} -> {cnt}\n")

                f.write("\n\n")

                results[key].append(result)
                global_run_id += 1

    print(f"\n=== SEZIONE {section_id}/{n_sections} COMPLETATA ===")
    print(f"Log salvati in: {output_file}")

    # Salvataggio pickle parziale
    #pickle_file = f"stopping_criteria_results_section_{section_id}.pkl"
    #with open(pickle_file, "wb") as f:
    #    pickle.dump(results, f)
    #print(f"Risultati salvati in: {pickle_file}")

    return results


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
    std_fitness = np.array([it["std_fitness"] for it in logger.iterations_data])  # Deviazione standard

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
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.show()


def plot_fitness_boxplots(results: Dict, title: str = "Fitness Distribution"):
    """
    Genera box plot delle distribuzioni fitness.

    HINT: Un boxplot per ogni configurazione testata
    """
    # TODO: Implementare

    # Estrazione delle configurazioni 
    configurations = sorted(results.keys())
    data_to_plot = []

    # Estrazione degli estremi della fitness per ogni configurazione 
    for configuration in configurations:
        runs  = results[configuration]
        gbest_values = []

        for run in runs:
            logger = run["logger"]
            gbest_iter = [it["gbest_fitness"] for it in logger.iterations_data]

            gbest_values.append(min(gbest_iter))
            gbest_values.append(max(gbest_iter))

        data_to_plot.append(gbest_values)

    # Plotting
    labels = [str(cfg) for cfg in configurations]
    plt.figure(figsize=(15, 6))
    plt.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Configurazione parametri")
    plt.ylabel("GBest Fitness (min e max per run)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_frequency(results: dict, top_k: int = 20):
    """
    Genera un bar graph per ogni swarm size con le top-k feature
    più frequentemente selezionate in media sui run.

    Args:
        results: dizionario dei risultati salvati (per swarm size)
        top_k: numero di feature da mostrare
    """
    for swarm_size, runs in results.items():
        # Aggrego le frequenze di selezione per tutte le run
        aggregated_counts = {}
        for run in runs:
            feature_counts = run["logger"].feature_counts
            for feat_idx, count in feature_counts.items():
                aggregated_counts[feat_idx] = aggregated_counts.get(feat_idx, 0) + count

        # Calcolo la media su tutte le run
        n_runs = len(runs)
        avg_counts = {feat: aggregated_counts[feat] / n_runs for feat in aggregated_counts}

        # Ordino le feature per frequenza media decrescente
        sorted_feats = sorted(avg_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Estrazione dei dati per il plot
        feature_indices = [f"F{feat}" for feat, _ in sorted_feats]
        frequencies = [freq for _, freq in sorted_feats]

        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(frequencies)), frequencies, color='skyblue')
        plt.xticks(range(len(frequencies)), feature_indices, rotation=45, ha='right')
        plt.xlabel("Feature")
        plt.ylabel("Frequenza media di selezione")
        plt.title(f"Top {top_k} Features - Swarm Size {swarm_size}")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_swarm_behavior(swarm_data: Dict, title: str = "Swarm Behavior"):
    """
    Visualizza il comportamento dello sciame nel tempo.

    HINT:
    - Dispersione delle particelle
    - Velocità media
    - Evoluzione del gbest
    """
    # TODO: Implementare

    logger = swarm_data["logger"]

    # Estrazione delle grandezze di interesse 
    iters = np.array([it["iteration"] for it in logger.iterations_data])
    dispersion = np.array([it["dispersion"] for it in logger.iterations_data])
    avg_vel = np.array([it["avg_velocity"] for it in logger.iterations_data])
    gbest = np.array([it["gbest_fitness"] for it in logger.iterations_data])

    # Plotting 

    fig, (ax1, ax2,ax3) = plt.subplots(1,3,figsize=(12, 5))

    # Plotting
    ax1.plot(iters, dispersion, color='orange', linewidth=2, label='Swarm Dispersion')
    # Plot del Best
    ax2.plot(iters, gbest, color='blue', linewidth=2.5, label='Global Best')

    # Linea ottima
    ax3.plot(iters, avg_vel, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Average Velocity')

    # Formattazione
    ax1.set_xlim(0, max(iters))
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('-')
    ax1.set_title(title+"\nDispersion")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlim(0, max(iters))
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('-')
    ax2.set_title(title+"\nGlobal Best")
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    ax3.set_xlim(0, max(iters))
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('-')
    ax3.set_title(title+"\nAverage Velocity")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.show()


def plot_all_swarms_convergence(aggregated_results_by_swarm: dict,
                                title: str = "Convergence Comparison Across Swarm Sizes"):
    """
    Plotta sullo stesso grafico:
    - Swarm Average (media su 30 run)
    - Global Best (media su 30 run)
    per tutte le swarm size.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(13, 6))

    # Target "optimum" = miglior fitness osservata globalmente
    global_optimum = max(
        res["best_fitness"]
        for res in aggregated_results_by_swarm.values()
    )

    for swarm_size, results in aggregated_results_by_swarm.items():
        logger = results["logger"]

        iters = np.array([it["iteration"] for it in logger.iterations_data])
        avg_fitness = np.array([it["avg_fitness"] for it in logger.iterations_data])
        gbest_fitness = np.array([it["gbest_fitness"] for it in logger.iterations_data])

        # Swarm Average (tratteggiata)
        ax.plot(
            iters,
            avg_fitness,
            linestyle="--",
            linewidth=2,
            label=f"Swarm Avg (S={swarm_size})"
        )

        # Global Best (continua)
        ax.plot(
            iters,
            gbest_fitness,
            linewidth=2.5,
            label=f"Global Best (S={swarm_size})"
        )

    # Target Optimum
    ax.axhline(
        global_optimum,
        color="black",
        linestyle=":",
        linewidth=2,
        label="Target Optimum (Best observed)"
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title(title)

    ax.legend(loc="lower right", ncol=2)

    ax.grid(True, alpha=0.3)

    ax.set_xlim(0, iters.max())

    plt.tight_layout()
    plt.show()


def plot_all_coeff_variations(aggregated_results_by_swarm: dict,
                              index_coeff: int,
                              title: str = "Dispersion - Fitness - Average Velocità"
                                ):
    """
    Plotta su 3 grafici:
    - Dispersione di default vs variazioni 
    - Fitness globale migliore di default vs variazioni 
    - Velocità media di default vs variaizoni 
    Si ha la stampa dei 3 grafici in 3 finestre diverse per tenere fermi due parametri e variarne uno in ciascuna finestra 
    """

    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(13, 6))

    # Target "optimum" = miglior fitness osservata globalmente
    global_optimum = max(
        res["best_fitness"]
        for res in aggregated_results_by_swarm.values()
    )
    if_cond=[[0.7],[2.0],[2.0]]

    if index_coeff == 1:
        if_cond[0]=[0.4,0.6,0.7,0.9]
    elif index_coeff ==2:
        if_cond[1]=[1.0,1.5,2.0,2.5]
    elif index_coeff ==3:
        if_cond[2]=[1.0,1.5,2.0,2.5]

    # Deault: (0.7,2.0,2.0)
    for coeff, results in aggregated_results_by_swarm.items():
       if (
                coeff[0] in if_cond[0] and
                coeff[1] in if_cond[1] and
                coeff[2] in if_cond[2]
            ):
            logger = results["logger"]

            iters = np.array([it["iteration"] for it in logger.iterations_data])
            dispersion = np.array([it["dispersion"] for it in logger.iterations_data])
            avg_vel = np.array([it["avg_velocity"] for it in logger.iterations_data])
            gbest = np.array([it["gbest_fitness"] for it in logger.iterations_data])

            # Dispersione
            ax1.plot(
                iters,
                dispersion,
                linestyle="-",
                linewidth=1.5,
                label=f"Dispersion -{coeff}"
            )

            # Global Best (continua)
            ax2.plot(
                iters,
                gbest,
                linewidth=1.5,
                label=f"Global Best -{coeff})"
            )
            
            # Velocità media 
            ax3.plot(
                iters,
                avg_vel,
                linewidth=1.5,
                label=f"Average velocity -{coeff})"
            )

        

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Dispersion")
    ax1.set_title("Swarm Behaviour - Dispersion")

    ax1.legend(loc="center right")

    ax1.grid(True, alpha=0.3)

    ax1.set_xlim(0, iters.max())

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Swarm Behaviour - Global Fitness")
    ax2.set_title(title)

    ax2.legend(loc="lower right")

    ax2.grid(True, alpha=0.3)

    ax2.set_xlim(0, iters.max())


    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Swarm Behaviour - Average Velocity")
    ax3.set_title(title)

    ax3.legend(loc="lower right")

    ax3.grid(True, alpha=0.3)

    ax3.set_xlim(0, iters.max())

    
    plt.show()

def print_top_features(results: dict, top_k: int = 20):
    """
    Stampa le top-k features più frequentemente selezionate in media
    per ogni swarm size.

    Args:
        results: dizionario dei risultati salvati (per swarm size)
        top_k: numero di feature da mostrare
    """
    for swarm_size, runs in results.items():
        print(f"\n=== Swarm Size: {swarm_size} ===")

        # Aggrego le frequenze di selezione per tutte le run
        aggregated_counts = {}

        for run in runs:
            feature_counts = run["logger"].feature_counts
            for feat_idx, count in feature_counts.items():
                aggregated_counts[feat_idx] = aggregated_counts.get(feat_idx, 0) + count

        # Calcolo la media su tutte le run
        n_runs = len(runs)
        avg_counts = {feat: aggregated_counts[feat] / n_runs for feat in aggregated_counts}

        # Ordino le feature per frequenza media decrescente
        sorted_feats = sorted(avg_counts.items(), key=lambda x: x[1], reverse=True)

        # Stampiamo le top-k
        print(f"Top {top_k} features più selezionate (in media):")
        for i, (feat_idx, avg_count) in enumerate(sorted_feats[:top_k], 1):
            print(f"{i:02d}. Feature {feat_idx} -> selezionata in media {avg_count:.2f} volte per run")


# =============================================================================
# FUNZIONI AUSILIARIE
# =============================================================================
def merge_stopping_criteria_results(n_sections: int = 1, output_file: str = "stopping_criteria_results_full.pkl"):
    """
    Unisce tutti i pickle generati da più sezioni in un unico dizionario.
    """

    final_results = {}

    # Combinazione dei vari pickle
    for section_id in range(1, n_sections + 1):
        pickle_file = f"stopping_criteria_results_section_{section_id}.pkl"
        try:
            with open(pickle_file, "rb") as f:
                partial_results = pickle.load(f)
                final_results.update(partial_results)
                print(f"Sezione {section_id} caricata, {len(partial_results)} combinazioni")
        except FileNotFoundError:
            print(f"Attenzione: file {pickle_file} non trovato, salto.")

    # Salvataggio finale
    with open(output_file, "wb") as f:
        pickle.dump(final_results, f)
    print(f"Tutti i risultati uniti in {output_file}")

    return final_results


def build_aggregated_results_for_plot(runs: list):
    """
    Costruisce un dizionario 'results' compatibile con
    plot_convergence_curves, aggregando più run.
    """

    n_runs = len(runs)
    n_iter = len(runs[0]["logger"].iterations_data)

    aggregated_logger = deepcopy(runs[0]["logger"])
    aggregated_logger.iterations_data = []

    # Calcolo delle grandezze di interesse medie su tutte le run 
    for i in range(n_iter):
        avg_fitness = np.mean([
            run["logger"].iterations_data[i]["avg_fitness"]
            for run in runs
        ])

        std_fitness = np.mean([
            run["logger"].iterations_data[i]["std_fitness"]
            for run in runs
        ])

        gbest_fitness = np.mean([
            run["logger"].iterations_data[i]["gbest_fitness"]
            for run in runs
        ])

        dispersion = np.mean([
            run["logger"].iterations_data[i]["dispersion"]
            for run in runs
        ])

        avg_velocity = np.mean([
            run["logger"].iterations_data[i]["avg_velocity"]
            for run in runs
        ])
        aggregated_logger.iterations_data.append({
            "iteration": i,
            "avg_fitness": avg_fitness,
            "std_fitness": std_fitness,
            "gbest_fitness": gbest_fitness,
            "dispersion" : dispersion,
            "avg_velocity" : avg_velocity
        })

    aggregated_results = {
        "logger": aggregated_logger,
        "best_fitness": max(
            run["best_fitness"] for run in runs
        )
    }

    return aggregated_results


# =============================================================================
# MAIN
# =============================================================================
def oldmain():
    # 1. Caricamento dati
    #X, y = load_darwin_dataset("DARWIN.csv")

    # 2. Conversione target
    #y_numeric = y.map({'P': 1, 'H': 0})

    # 3. Esecuzione dell'esperimento pesante
    #print("Avvio dell'esperimento Swarm Size (potrebbe richiedere tempo)...")

    # Nota: Assicurati di passare i parametri corretti che la funzione si aspetta
    #results = run_experiment_swarm_size(X, y_numeric, n_runs=30)

    # 4. SALVATAGGIO DEI RISULTATI (PICKLE)
    #filename = "risultati_swarm_size.pkl"
    #print(f"Salvataggio dei risultati in {filename}...")

    #with open(filename, "wb") as f:
    #    pickle.dump(results, f)

    #print("Salvataggio completato")

    # 1. Caricamento risultati
    with open("risultati_swarm_size.pkl", "rb") as f:
        results = pickle.load(f)

    # 2. Costruzione logger aggregati per ogni swarm size
    aggregated_by_swarm = {}

    for swarm_size, runs in results.items():
        aggregated_by_swarm[swarm_size] = build_aggregated_results_for_plot(runs)
        print(f"Plot convergenza media - Swarm size {swarm_size}")

        aggregated_results = build_aggregated_results_for_plot(runs)

        plot_convergence_curves(
            aggregated_results,
            title=f"Convergence Curve (Mean of 30 runs) - Swarm size {swarm_size}"
        )

    # 3. Plot unico con TUTTE le swarm size
    plot_all_swarms_convergence(
        aggregated_by_swarm,
        title="PSO Convergence Comparison (Mean over 30 runs)"
    )

def main():

    # ============================================================
    # MODALITÀ: RUN ESPERIMENTO
    # ============================================================

    if MODE == "run":

        print("Caricamento dataset...")
        X, y = load_darwin_dataset(DATASET_PATH)
        y_numeric = y.map({'P': 1, 'H': 0})

        if EXPERIMENT == "swarm":
            print("Avvio esperimento: SWARM SIZE")
            results = run_experiment_swarm_size(X, y_numeric, n_runs=N_RUNS)

        elif EXPERIMENT == "coeff":
            print("Avvio esperimento: PSO COEFFICIENTS")
            results = run_experiment_pso_coefficients(X, y_numeric, n_runs=N_RUNS, section_id=1, n_sections=1)

        elif EXPERIMENT == "stop":
            print("Avvio esperimento: STOPPING CRITERIA")
            #results = run_experiment_stopping_criteria(X, y_numeric, n_runs=N_RUNS)
            results = run_experiment_stopping_criteria(X, y_numeric, n_runs=30, section_id=1, n_sections=1)

        else:
            raise ValueError("Esperimento non riconosciuto")

        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(results, f)

        print(f"Risultati salvati in {PICKLE_FILE}")

    # ============================================================
    # MODALITÀ: PLOT RISULTATI
    # ============================================================

    elif MODE == "plot":

        if EXPERIMENT == "swarm":
            print(f"Caricamento risultati da {PICKLE_FILE}")
            try:
                with open(PICKLE_FILE, "rb") as f:
                    results = pickle.load(f)
            except FileNotFoundError:
                print(f"Errore: il file {PICKLE_FILE} non esiste.")
                results = None
                sys.exit(1)
            except Exception as e:
                print(f"Errore durante il caricamento del file: {e}")
                results = None
                sys.exit(1)

            # Aggregazione per swarm size

            plot_fitness_boxplots(results)

            aggregated_by_swarm = {}

            for swarm_size, runs in results.items():
                aggregated = build_aggregated_results_for_plot(runs)
                aggregated_by_swarm[swarm_size] = aggregated

                plot_convergence_curves(
                    aggregated,
                    title=f"Convergence Curve (Mean of {N_RUNS} runs) - Swarm size {swarm_size}"
                )
                aggregated_by_swarm[swarm_size] = aggregated
                plot_swarm_behavior(
                    aggregated,
                    title=f"Swarm behavior (Mean of {N_RUNS} runs) - Swarm size {swarm_size}"
                )
            # Plot comparativo finale
            plot_all_swarms_convergence(
                aggregated_by_swarm,
                title="PSO Convergence Comparison (Mean over runs)"
            )
            # Plot della frequenza di scelta delle feature 
            plot_feature_frequency(results, top_k=100)

        elif EXPERIMENT == "coeff":

            print(f"Caricamento risultati da {PICKLE_FILE}")
            try:
                with open(PICKLE_FILE, "rb") as f:
                    results = pickle.load(f)
            except FileNotFoundError:
                print(f"Errore: il file {PICKLE_FILE} non esiste.")
                results = None
                sys.exit(1)
            except Exception as e:
                print(f"Errore durante il caricamento del file: {e}")
                results = None
                sys.exit(1)

            # Aggregazione per swarm size

            plot_fitness_boxplots(results)

            aggregated_by_coeff = {}

            for coefficients, runs in results.items():
                aggregated = build_aggregated_results_for_plot(runs)
                aggregated_by_coeff[coefficients] = aggregated

                """ plot_convergence_curves(
                    aggregated,
                    title=f"Convergence Curve (Mean of {N_RUNS} runs) - Coefficients {coefficients}"
                )
                aggregated_by_coeff[coefficients] = aggregated
                plot_swarm_behavior(
                    aggregated,
                    title=f"Swarm behavior (Mean of {N_RUNS} runs) - Coefficients {coefficients}"
                )
 """
            # Plot comparativo finale
            plot_all_swarms_convergence(
                aggregated_by_coeff,
                title="PSO Convergence Comparison (Mean over runs)"
            )
            for i in range(1,4):
                plot_all_coeff_variations(
                    aggregated_by_coeff,
                    title="PSO Behaviour Comparison (Mean over runs)",
                    index_coeff=i
                )
            # Plot della frequenza di scelta delle feature 
            plot_feature_frequency(results, top_k=100)


        elif EXPERIMENT == "stop":
            
            print(f"Caricamento risultati da {PICKLE_FILE}")
            try:
                with open(PICKLE_FILE, "rb") as f:
                    results = pickle.load(f)
            except FileNotFoundError:
                print(f"Errore: il file {PICKLE_FILE} non esiste.")
                results = None
                sys.exit(1)
            except Exception as e:
                print(f"Errore durante il caricamento del file: {e}")
                results = None
                sys.exit(1)

            # Aggregazione per swarm size

            plot_fitness_boxplots(results)

            aggregated_by_stop = {}

            for stop, runs in results.items():
                aggregated = build_aggregated_results_for_plot(runs)
                aggregated_by_stop[stop] = aggregated

                plot_convergence_curves(
                    aggregated,
                    title=f"Convergence Curve (Mean of {N_RUNS} runs) - Coefficients {stop}"
                )
                aggregated_by_stop[stop] = aggregated
                plot_swarm_behavior(
                    aggregated,
                    title=f"Swarm behavior (Mean of {N_RUNS} runs) - Coefficients {stop}"
                )
                
            # Plot comparativo finale
            plot_all_swarms_convergence(
                aggregated_by_stop,
                title="PSO Convergence Comparison (Mean over runs)"
            )
            
            # Plot della frequenza di scelta delle feature 
            plot_feature_frequency(results, top_k=100)
            

         

    else:
        raise ValueError("MODE deve essere 'run' o 'plot'")


if __name__ == "__main__":

    # ============================================================
    # CONFIGURAZIONE
    # ============================================================

    MODE = "plot"          # "run" | "plot"
    EXPERIMENT = "coeff"   # "swarm" | "coeff" | "stop"

    N_RUNS = 30
    DATASET_PATH = "DARWIN.csv"
    PICKLE_FILE = f"pickles/risultati_{EXPERIMENT}.pkl"

    main()
