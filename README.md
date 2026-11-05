# Informed Split Tensorized SPN (IST-SPN)

## Aufbau des Repositories

```
.
├── src/
│   └── ist.py          # InformedSplitTensorizedSPN
├── experiments/
│   ├── exp1.py         # Cancer Datensatz (30 Features)
│   ├── exp2.py         # NCC Datensatz
│   └── utils.py        # Hilfsfunktionen (Seed, Daten, Auswertung)
```

## Aufbau des IST-SPN (`src/ist.py`)

### Schritt 1: Struktur bestimmen

Ausgehend von den Daten wird die Partitionierung bestimmt,
z.B. Datensatz mit 8 Features => `[[[1,3],[2,4]], [[5,8],[6,7]]]`.

Pro Ebene wird das Feature Set für jeden Teilbaum rekursiv in `determine_split` aufgeteilt:

- **Basisfall:** Maximale Tiefe ist erreicht, gebe Blatt zurück => `[1,3]`
- **Rekursionsfall:**
  1. Score Matrix bestimmen (`dependency_test` = `gtest` oder `rdc`)
  2. Features in Teilbäume sortieren:
     `[1,2,3,4]` + Score Matrix => `[Rekursionsaufruf mit (1,3), Rekursionsaufruf mit (2,4)]`

### Varianten

**Abhängigkeitstest (`dependency_test`)**

| Wert | Beschreibung |
|---|---|
| `gtest` | G-Test auf Basis der Features, die am Median in 0/1 umgewandelt werden |
| `rdc` | Randomized Dependence Coefficient. Bei mehr als 5000 Zeilen wird eine Stichprobe verwendet |

**Aufteilung (`ordering`)**

| Wert | Beschreibung |
|---|---|
| `heuristic` | Greedy: Paare mit dem höchsten Score werden zuerst in denselben Branch gelegt, die Branches bleiben ausgeglichen |
| `optimal` | Balancierter Min-Cut: Die Summe der Scores zwischen den beiden Branches wird minimiert. Start mit spektraler Aufteilung (Fiedler-Vektor), danach Verbesserung durch Tauschen einzelner Features |
| `random` | Zufällige Aufteilung in zwei gleich große Hälften, dient als Vergleich. Der Abhängigkeitstest wird hier nicht benutzt |

### Schritt 2: RAT-SPN bauen

Nutze den Plan, um den SPN zu bauen:
`[[[1,3],[2,4]], [[5,8],[6,7]]]` => `Sum(Product(Sum(Product(Factorization))))`

`create_spn` ist eine Kopie der gleichnamigen Funktion aus dem RAT-SPN in SPFlow. Es gibt nur einen Unterschied: Die Klasse `FixedFactorize` ersetzt `Factorize`.

Das normale `Factorize` verteilt die Features zufällig auf die Gruppen. `FixedFactorize` baut stattdessen die Zuordnungsmatrix aus den Blättern des Plans aus Schritt 1. Zeilen sind Features, Spalten sind Gruppen. Für `[[1,3],[2,4]], [[5,8],[6,7]]` ergibt das:

```
           G1 G2 G3 G4
Feature 1   1  0  0  0
Feature 2   0  1  0  0
Feature 3   1  0  0  0
Feature 4   0  1  0  0
Feature 5   0  0  1  0
Feature 6   0  0  0  1
Feature 7   0  0  0  1
Feature 8   0  0  1  0
```

## Ergebnisse zu Experiment 1

### Mit Gradient Descent (`PYTHONPATH=. python experiments/exp1.py --run gd`)
<img width="655" height="567" alt="Pasted Graphic" src="https://github.com/user-attachments/assets/230d1dd5-5ec3-44dc-9c32-756edf599d0c" />

### Mit EM (`PYTHONPATH=. python experiments/exp1.py --run em`)
<img width="639" height="569" alt="Pasted Graphic 1" src="https://github.com/user-attachments/assets/b6543808-d962-4c9d-aa28-76606cb088d9" />

Alle Ansätze verhalten sich in etwa gleich, Optimal > Heuristik > Random.






