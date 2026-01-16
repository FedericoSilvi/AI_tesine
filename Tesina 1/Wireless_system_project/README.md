# Tesina 1 - Infrastruttura di Comunicazione Wireless

## Problema

Una startup di telecomunicazioni deve progettare un'infrastruttura di rete wireless per connettere **20 siti strategici** in una regione urbana/industriale, minimizzando i costi di installazione e massimizzando l'efficienza della connessione.

---

## Specifiche del Problema

- Rete composta da **20 nodi** di comunicazione
- Ogni nodo rappresenta un sito potenziale per stazioni ripetitori
- I costi di connessione variano in base a:
  - Distanza geografica
  - Caratteristiche del nodo (latenza, vulnerabilità, consumo energetico)
  - Investimento infrastrutturale

---

## Obiettivi

1. Implementare l'algoritmo per ottenere il **Minimum Spanning Tree (MST)**
2. Utilizzare **Kruskal's algorithm** o **Prim's algorithm**
3. Ottimizzare la connessione minimizzando:
   - Costi di installazione
   - Lunghezza totale dei cavi/connessioni
   - Impatto sulle performance

---

## Vincoli

- Tutti i 20 nodi devono essere connessi
- Considerare le caratteristiche specifiche di ogni scenario
- Gestire differenti tipologie di costi di connessione

---

## Fasi del Progetto

### Fase 1: Sviluppo dell'Algoritmo

Implementare MST con criteri personalizzati. Considerare:
- Costi di connessione
- Qualità del segnale
- Caratteristiche dello scenario

### Fase 2: Simulazione degli Scenari

---

#### 🌆 Scenario Smart City IoT

| Parametro | Valore |
|-----------|--------|
| **Obiettivo** | Connettere sensori IoT in ambiente urbano |
| **Max latenza per hop** | 50ms |
| **Fattore banda** | 1.5 |
| **Vincoli** | Minimizzare latenza, considerare requisiti di banda |

**Tipi di nodi**:
- Sensori traffico (alta priorità, colore rosso)
- Sensori ambientali (media priorità, colore arancione)
- Luci smart (bassa priorità, colore blu)

---

#### 🌋 Scenario Zona Sismica

| Parametro | Valore |
|-----------|--------|
| **Obiettivo** | Garantire ridondanza di comunicazione |
| **Fattore ridondanza** | 2.5 |
| **Max vulnerabilità** | 0.65 |
| **Vincoli** | Massimizzare affidabilità connessioni |

**Soglie vulnerabilità**: Alta (>0.85), Media (0.55), Base (0.35)

---

#### ⚡ Scenario Ottimizzazione Energetica

| Parametro | Valore |
|-----------|--------|
| **Obiettivo** | Minimizzare consumo energetico |
| **Max potenza per nodo** | 85W |
| **Budget totale potenza** | 1350W |
| **Vincoli** | Limitare numero ripetitori |

---

## Struttura della Rete

### Aree e Caratteristiche
| Area | Nodi | Priorità/Caratteristica |
|------|------|-------------------------|
| **Centro** | 0, 1, 2, 3 | Media priorità |
| **Nord** | 4, 5, 6, 7 | Alta priorità (sensori traffico) |
| **Sud** | 8, 9, 10, 11 | Bassa priorità (smart lights) |
| **Est** | 12, 13, 14, 15 | Media priorità |
| **Ovest** | 16, 17, 18, 19 | Media priorità |

---

## Output Richiesti per Ogni Scenario

- ✅ Albero di connessione minimo (MST)
- 📊 Costo totale dell'infrastruttura
- 🔍 Analisi dei nodi critici
- 🔄 Confronto tra diversi algoritmi MST (Kruskal vs Prim)
- 📈 Visualizzazione grafica della rete

---

## Criteri di Valutazione

- Correttezza implementazione **MST** (Kruskal/Prim)
- Efficacia delle funzioni di costo personalizzate
- Performance delle soluzioni calcolate
- Capacità di adattamento a scenari diversi

---

## File da Implementare

### Algoritmi MST
- `src/algorithm/kruskal.py` - Implementare la classe `KruskalMST`
- `src/algorithm/prim.py` - Implementare la classe `PrimMST`

### Scenari
- `src/scenarios/smartcity.py` - Implementare `solve_smartcity_scenario()`
- `src/scenarios/seismic.py` - Implementare `solve_seismic_scenario()`
- `src/scenarios/energy.py` - Implementare `solve_energy_scenario()`

### Funzioni di Costo (Opzionale)
- `src/algorithm/cost_functions.py` - Estendere le classi di costo per ottimizzazioni avanzate

---

## Da Fornire prima dell'Orale

- 📁 **Codice completo** (repository GitHub)
- 📝 **Report** con plot, scelte adottate e motivazioni

---

## Esecuzione

```bash
# Scenario Smart City IoT
python main.py --scenario smartcity

# Scenario Zona Sismica
python main.py --scenario seismic

# Scenario Ottimizzazione Energetica
python main.py --scenario energy

# Scegliere algoritmo (default: kruskal)
python main.py --scenario smartcity --algorithm prim

# Aiuto
python main.py --help
```

---

## Struttura del Progetto

```
Wireless_system_project/
├── README.md                     # Questo file
├── main.py                       # Entry point principale
├── requirements.txt              # Dipendenze Python
└── src/
    ├── algorithm/
    │   ├── kruskal.py           # TODO: Implementare Kruskal MST
    │   ├── prim.py              # TODO: Implementare Prim MST
    │   └── cost_functions.py    # Funzioni di costo personalizzabili
    ├── models/
    │   ├── network.py           # Classe WirelessNetwork (fornita)
    │   └── node.py              # Classe Node (fornita)
    ├── scenarios/
    │   ├── smartcity.py         # TODO: Implementare scenario Smart City
    │   ├── seismic.py           # TODO: Implementare scenario sismico
    │   └── energy.py            # TODO: Implementare scenario energia
    └── utils/
        └── visualization.py     # Utilità di visualizzazione (fornita)
```

---

## Dipendenze

Installare le dipendenze con:
```bash
pip install -r requirements.txt
```

Le dipendenze principali sono:
- `networkx` - Per la gestione dei grafi
- `matplotlib` - Per la visualizzazione
- `numpy` - Per i calcoli numerici