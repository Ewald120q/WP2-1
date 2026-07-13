# Joint Cascade MoE

Dieses Paket trainiert `f_small`, `f_mid`, `f_large`, `r1` und `r2`
in einem gemeinsamen PyTorch-Graphen.

## Eingabedomänen

Das Ensemble erhält immer das vollständige Batch-Dictionary. Die bereits
vorhandene `mode`-Konfiguration der Experten wählt daraus die passende Domäne:

- `f_small`, `mode="dmt"`: `batch["dm_time"]`
- `f_mid`, `mode="ft"`: `batch["freq_time"]`
- `f_large`, `mode="dmft"`: beide Repräsentationen als zwei Kanäle

Der Dataset-Loader wird deshalb immer mit `use_freq_time=True` erstellt.

## Dateien

- `joint_ensemble.py`: weicher Trainingsforward und sparse harte Inferenz
- `loss.py`: Ensemble-, Kosten- und Experten-Loss
- `train_helper.py`: gemeinsame Trainings- und Validierungsfunktionen
- `checkpoints.py`: Laden alter Experten-/Rejector-Checkpoints
- `train_joint_ensemble.py`: ausführbarer Einstiegspunkt
- `config.example.json`: vollständige Beispielkonfiguration
- `randomsearch_joint_ensemble.py`: mehrere zufällige Hyperparameter-Trials

## Start

Aus `single_pulse_classifier_training`:

```bash
cp moe/config.example.json moe/config.local.json
python -m moe.train_joint_ensemble --config moe/config.local.json
```

Alternativ funktioniert auch:

```bash
python moe/train_joint_ensemble.py --config moe/config.local.json
```

`config.local.json` bleibt durch das globale `*.json`-Ignore lokal. Nur die
versionierten `*.example.json`-Dateien sind explizit vom Ignore ausgenommen.

## Random Search

Die normale Trainingsconfig enthält zusätzlich den Abschnitt `random_search`.
Pro Trial werden die dort definierten Werte überschrieben:

```bash
cp moe/config.example.json moe/config.local.json
```

```bash
python -m moe.randomsearch_joint_ensemble \
  --config moe/config.local.json \
  --worker-id 0
```

Unterstützte Verteilungen:

- `uniform`: Fließkommawert zwischen `min` und `max`
- `log_uniform`: logarithmisch gezogener positiver Fließkommawert
- `choice`: zufälliger Eintrag aus `values`

Die Schlüssel unter `parameters` sind Pfade in der normalen Trainingsconfig,
beispielsweise `loss.lambda_cost`. Das optionale Feld `name` legt die kurze
Bezeichnung im Ordnernamen fest. Run-Namen enthalten Worker, Trial, Seed sowie
alle gezogenen Hyperparameter.

Mehrere Prozesse können mit unterschiedlichen Worker-IDs gestartet werden:

```bash
python -m moe.randomsearch_joint_ensemble --config moe/config.local.json --worker-id 0
python -m moe.randomsearch_joint_ensemble --config moe/config.local.json --worker-id 1
```

Jeder Worker erhält einen getrennten Seed-Bereich. Jeder Run speichert seine
gezogene Config und ein `randomsearch_result.json`. Bereits vorhandene
Run-Ordner werden übersprungen.

## Initialisierung

Die `checkpoint`-Felder können `null` bleiben, um vollständig neu zu trainieren.
Für eine stabilere erste Joint-Training-Runde sollten dort die bestehenden
Experten- und Rejector-Checkpoints eingetragen werden. Historische
`EmbeddingRejector`-Checkpoints mit `1.*`-Keys werden automatisch auf den
reinen Rejector-Head abgebildet.

## Softes Training

Für ein Sample gelten:

```text
w_small = 1 - q1
w_mid   = q1 * (1 - q2)
w_large = q1 * q2
```

Die drei normalisierten Expertenverteilungen werden mit diesen Gewichten
gemischt. `forward()` liefert gemischte Log-Wahrscheinlichkeiten; der
Trainingsloop verwendet deshalb `NLLLoss`.

Alle Experten laufen während des Trainings auf dem gesamten Batch. Das ist
notwendig, damit alle Pfade differenzierbar bleiben. Der Kosten-Loss modelliert
die erwarteten Kosten der späteren harten Inferenz, nicht die tatsächlichen
Trainingskosten.

## Harte Inferenz

`predict_hard()` führt `f_mid` nur auf von R1 weitergeleiteten Samples aus und
`f_large` nur auf von R2 weitergeleiteten Samples. Dabei werden alle Tensoren im
Batch-Dictionary gemeinsam geschnitten, sodass DM-Time und Frequency-Time
synchron bleiben.

## Loss

Der gemeinsame Loss ist:

```text
L = L_ensemble + lambda_cost * L_cost + alpha_experts * L_experts

L_ensemble = CE(p_mix, y)
L_cost     = w_small*c_small + w_mid*c_mid + w_large*c_large
L_experts  = CE(p_small,y) + CE(p_mid,y) + CE(p_large,y)
```

Die drei Latenzen werden in `config.example.json` eingetragen. Der Loss teilt
sie intern durch die größte Latenz, sodass alle Kosten zwischen null und eins
liegen. Die Werte in der Beispielkonfiguration sind nur Platzhalter.

Für reale Kaskadenkosten können `c_small`, `c_mid` und `c_large` als gesamte
Latenz des jeweils endenden Pfades verstanden werden: bis Small, bis Mid und bis
Large.
