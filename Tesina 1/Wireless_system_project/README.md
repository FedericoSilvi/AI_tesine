# Tesina 1 - Infrastruttura di Comunicazione Wireless

## Problema

Una startup di telecomunicazioni deve progettare un'infrastruttura di rete wireless per connettere **20 siti strategici** in una regione montuosa, minimizzando i costi di installazione e massimizzando l'efficienza della connessione.

---

## Specifiche del Problema

- Rete composta da **20 nodi** di comunicazione
- Ogni nodo rappresenta un sito potenziale per stazioni ripetitori
- I costi di connessione variano in base a:
  - Distanza geografica
  - Difficoltà di terreno
  - Investimento infrastrutturale

---

## Obiettivi

1. Implementare l'algoritmo per ottenere il **Minimum Spanning Tree (MST)**
2. Utilizzare **Kruskal's algorithm** o **Prim's algorithm**
3. Ottimizzare la connessione minimizzando:
   - Costi di installazione
   - Lunghezza totale dei cavi/connessioni
   - Impatto ambientale

---

## Vincoli

- Tutti i 20 nodi devono essere connessi
- Considerare le caratteristiche del territorio
- Gestire differenti tipologie di costi di connessione

---

## Fasi del Progetto

### Fase 1: Sviluppo dell'Algoritmo

Implementare MST con criteri personalizzati. Considerare:
- Costi di connessione
- Qualità del segnale
- Difficoltà di installazione

### Fase 2: Simulazione degli Scenari

---

#### 🏔️ Scenario Montagna

| Parametro | Valore |
|-----------|--------|
| **Obiettivo** | Connettere siti in zona alpina |
| **Max distanza link** | 850 |
| **Fattore elevazione** | 1.8 |
| **Vincoli** | Minimizzare lunghezza cavi, considerare dislivelli |

**Elevazioni nodi montagna**: 950m, 900m, 820m, 650m

---

#### 🌋 Scenario Zona Sismica

| Parametro | Valore |
|-----------|--------|
| **Obiettivo** | Garantire ridondanza di comunicazione |
| **Fattore ridondanza** | 2.5 |
| **Max vulnerabilità** | 0.65 |
| **Vincoli** | Massimizzare affidabilità connessioni |

**Soglie vulnerabilità**: Alta (>0.85 elevazione >600m), Media (0.55 elevazione <40m), Base (0.35)

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

### Aree e Elevazioni
| Area | Nodi | Elevazione (m) |
|------|------|----------------|
| **Montagna (Nord)** | 4, 5, 6, 7 | 950, 900, 820, 650 |
| **Valle (Sud)** | 8, 9, 10, 11 | 35, 25, 30, 15 |
| **Centro** | 0, 1, 2, 3 | 120, 180, 140, 110 |
| **Est** | 12, 13, 14, 15 | 220, 280, 350, 320 |
| **Ovest** | 16, 17, 18, 19 | 200, 250, 300, 270 |

---

## Output Richiesti per Ogni Scenario

- ✅ Albero di connessione minimo
- 📊 Costo totale dell'infrastruttura
- 🔍 Analisi dei nodi critici
- 🔄 Confronto tra diversi algoritmi MST
- 📈 Visualizzazione grafica della rete

---

## Criteri di Valutazione

- Correttezza implementazione **MST** (Kruskal/Prim)
- Efficacia delle funzioni di costo personalizzate
- Performance delle soluzioni calcolate
- Capacità di adattamento a scenari diversi

---

## Da Fornire prima dell'Orale

- 📁 **Codice completo** (repository GitHub)
- 📝 **Report** con plot, scelte adottate e motivazioni

---

## Esecuzione

```bash
# Scenario Montagna
python main.py --scenario mountain

# Scenario Zona Sismica
python main.py --scenario seismic

# Scenario Ottimizzazione Energetica
python main.py --scenario energy

# Scegliere algoritmo (default: kruskal)
python main.py --scenario mountain --algorithm prim

# Aiuto
python main.py --help
```