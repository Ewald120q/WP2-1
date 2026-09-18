Hi!

Kurzer Disclaimer: Der Code ist nicht super aufgeräumt. Ich hab viel legacy code gelöscht, kann aber leider nicht behaupten dass die ganze codebase aufs mindeste reduziert wurde.
Trotzdem ist alles wichtige enthalten. Außerdem hab ich für etwas mehr Ordnung Die Files in Ordner geschoben. Wenn ihr Code ausführt, müsst ihr also aufpassen dass die Pfade stimmen.
Die ReadMe ist zum großten Teil von Codex wo ich zwischendurch meinen Senf zu etwas abgebe :D

Das ganze Repo ist ein Fork von Andreis Codebase. Ich hab einzelne Dinge unter `DM_time_dataset_creator` geändert. Das allermeiste liegt in `single_pulse_classifier_training`. In `psrsigsim` hab ich nichts verändert.

# DM_time_dataset_creator
hier kann der Datensatz erstellt werden. Dafür müsst ihr zuerst mit TransientX eine Filterbank einlesen und Pulskandidaten erzeugen. Diese Pulskandidaten haben die Endungen `.cands`. Die müsst ihr mit dem Code in DM_time_dataset_creator einlesen um daraus ein Python/Pytorch Datensatz bauen zu können.

# psrsigsim
von andrei für so live simulation vom realtime classifier. hat bei mir nicht wirklich funktioniert. das baut auf super alten packages die man schwer zum laufen bekommt. kann auch sein dass das nur mit tensorflow modellen geht. hier ist also nichts interesantes.

# single_pulse_classifier_training
ich liste hier die ordner und files auf und schreibe zur orientierung was dazu:

- `finetune/`: Datensätze, Skripte und Notebooks zum Fine-Tuning von
  Klassifikatoren und Routing-Zielen.
- `rejector_analysis/`: Auswertung von R1/R2, Schwellenwerten und
  Precision-Recall-Kurven.
- `plot/`: Notebooks und erzeugte Abbildungen für Experimente und Arbeit.
- `hls/`: Prototyp mäßig Export und Evaluation quantisierter Modelle für HLS/FPGA. (hab das kurz probiert bis Sebastian meinte das wäre overkill, dann hab ich es schnell gelassen. wollte nur dass ihr seht dass ich daran gedacht habe)
- `moe/`: gemeinsames Training der Klassifikatoren und Rejector-Kaskade;
  Details stehen im nächsten Abschnitt.


- `DMTimeShardDataset.py`: PyTorch-Dataset/Loader für Shards und Manifeste.
- `ResNet.py`: allgemeine ResNet-Bausteine und -Architektur.
- `training_models_base.py`: gemeinsame Eingabeaufbereitung für DM-Time,
  Frequency-Time und die kombinierte Darstellung.
- `training_models.py`: Klassifikatorvarianten, ResNet-Modelle und
  Late-/Mid-Fusion-Modelle.
- `training_models_GAP.py`: kompakte CNN-Varianten mit Global Average Pooling.
- `embedding_processing_models.py`: Netze, die Feature-Maps für die Rejectoren
  weiterverarbeiten.
- `training.py`: regulärer Trainingsablauf für Klassifikatoren.
- `training_utils.py`: Konfiguration, Label-Encoding, Checkpoint- und
  Plot-Hilfsfunktionen.
- `_config.json`: Beispiel-/Arbeitskonfiguration für das reguläre Training.
- `gridsearch.py`, `gridsearch_rs.py`: Grid- bzw. Random-Search für
  Klassifikator-Hyperparameter.
- `ensemble.py`: klassische mehrstufige Rejection-Ensembles und
  Vorhersagehilfen.
- `rejection_ensemble_helper.py`: Auswertung, optimale Routen und SNR-Plots
  für diese Ensembles.
- `rejector.py`: PyTorch-Basis und Embedding-Varianten der Rejectoren.
- `skrejector.py`: einfacher SNR-basierter Decision-Tree-Rejector.
- `train_skrejector.py`: Experiment-/Trainingsskript für den sklearn-Rejector.
- `gridsearch_rejector.py`: Random Search für R1/R2-Architektur und
  Hyperparameter.
- `analyse_rejectors.ipynb`, `analyse_rejectors_finetune.ipynb`: interaktive
  Rejector-Auswertung vor bzw. nach Fine-Tuning.
- `benchmark_model_inference.ipynb`: Laufzeit-/Inferenz-Benchmark.
- `rejection_ensemble.ipynb`: explorative Analyse der Rejection-Ensembles.
- `evaluate_classificators.ipynb`: Auswertung der einzelnen Klassifikatoren.

### Dateien direkt in `single_pulse_classifier_training/moe/`

- `joint_ensemble.py`: Modellgraph, Routing und sparsame harte Inferenz.
- `loss.py`: gewichteter Ensemble-Loss sowie Zusatzterme für Experten/Routing.
- `train_helper.py`: Trainings-, Validierungs- und Checkpoint-Schleifen.
- `train_joint_ensemble.py`: ausführbarer Einstiegspunkt für das Joint Training.
- `randomsearch_joint_ensemble.py`: parallele Random-Search-Experimente.
- `checkpoints.py`: kompatibles Laden alter Experten-/Rejector-Gewichte und
  Speichern gemeinsamer Checkpoints.
- `config.json`: Arbeitskonfiguration für Daten, Experten, Rejectoren, Budget,
  Training und Inferenzschwellen.
- `evaluate_moe.ipynb`, `pr-auc-rejector_moe.ipynb`, `r1_r2_threshold.ipynb`:
  interaktive Auswertung der MoE und ihrer Rejectoren.
- `rejector_precision_recall.pdf`: exportierte Precision-Recall-Abbildung.


