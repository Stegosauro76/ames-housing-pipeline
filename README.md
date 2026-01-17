Ames Housing – End-to-End Machine Learning Evaluation Framework
Descrizione del progetto

Questo repository contiene un framework completo di Data Science e Machine Learning applicato al Ames Housing Dataset.
L’obiettivo è analizzare il dataset in modo sistematico e valutarne le prestazioni su tre diversi task di apprendimento automatico:

Regression: previsione del prezzo di vendita delle abitazioni

Clustering: individuazione di pattern e segmenti latenti nel mercato immobiliare

Classification: categorizzazione degli immobili in fasce di prezzo (Low / Medium / High)

Il progetto copre l’intero ciclo di vita di un’analisi di Data Science: dalla pulizia dei dati alla feature engineering, dalla selezione delle feature alla validazione dei modelli, fino alla sintesi finale dei risultati.

Dataset

Il dataset utilizzato è il Ames Housing Dataset, largamente impiegato in ambito accademico come alternativa più realistica al Boston Housing Dataset.

Caratteristiche principali:

Oltre 80 variabili descrittive per ogni abitazione

Feature numeriche e categoriche

Presenza di valori mancanti, outlier e variabili ad alta cardinalità

Target principale: SalePrice

Il dataset è incluso nella cartella data/ con il nome AmesHousing.csv.

Struttura del repository
├── data/
│   └── AmesHousing.csv
│
├── images/
│   ├── eda/
│   ├── feature_importance/
│   └── clustering/
│
├── src/
│   └── ames_housing_evaluator.py
│
├── report/
│   └── ames_housing_report.tex
│
├── README.md
└── requirements.txt


Descrizione:

data/: dataset utilizzato per l’analisi

images/: grafici e visualizzazioni (EDA, importanza delle feature, clustering)

src/: codice Python principale del progetto

report/: relazione tecnica in LaTeX (opzionale)

requirements.txt: dipendenze Python

Metodologia
Analisi esplorativa dei dati (EDA)

L’analisi iniziale include:

Distribuzione delle variabili

Tipi di dato e cardinalità

Analisi dei valori mancanti

Studio della distribuzione del target (SalePrice)

Valutazione di skewness e kurtosis

Preprocessing e Feature Engineering

Il preprocessing è progettato per essere robusto e prevenire il data leakage. Include:

Rimozione di colonne identificative

Creazione di nuove feature (es. TotalSF, HouseAge, SinceRemod)

Trasformazione logaritmica del target per ridurre la skewness

Imputazione dei valori mancanti (mediana per numeriche, categoria “Missing” per categoriche)

Gestione degli outlier tramite clipping basato su IQR

Encoding delle variabili categoriche:

One-Hot Encoding per bassa cardinalità

Frequency Encoding + Target Encoding Out-of-Fold per alta cardinalità

Rimozione di feature a varianza nulla

Standardizzazione delle feature numeriche

Feature Selection

La selezione delle feature viene effettuata tramite Mutual Information Regression, selezionando le k feature più informative rispetto al target log-trasformato.

Questa fase consente di:

Ridurre la dimensionalità

Migliorare la generalizzazione dei modelli

Aumentare la stabilità delle performance in cross-validation

Modelli e task di apprendimento
Regressione

Modello: Random Forest Regressor

Validazione: K-Fold Cross-Validation

Metriche:

R²

RMSE (scala logaritmica)

MAE (scala logaritmica)

Analisi delle feature più importanti

Clustering

Algoritmi utilizzati:

K-Means (principale)

DBSCAN (opzionale)

Riduzione dimensionale tramite PCA (85% della varianza spiegata)

Metriche di valutazione:

Silhouette Score

Davies–Bouldin Index

Classificazione

Target: fasce di prezzo (Low, Medium, High)

Modello: Random Forest Classifier

Validazione: K-Fold Cross-Validation

Metriche:

Accuracy

F1-score (weighted)

Precision

Recall

Confronto con baseline basata sulla classe più frequente

Risultati

I risultati mostrano che il dataset Ames Housing è particolarmente adatto a:

Regression: forte capacità predittiva grazie alla ricchezza delle feature

Classification: buona separabilità tra le fasce di prezzo

Clustering: struttura latente presente ma meno marcata rispetto ai task supervisionati

I dettagli quantitativi sono riportati nel log di esecuzione e, opzionalmente, nella relazione in LaTeX.

Requisiti

Il progetto richiede Python 3.8+ e le principali librerie di Data Science:

pandas

numpy

scikit-learn

scipy

matplotlib

seaborn

Per installare le dipendenze:

pip install -r requirements.txt

Esecuzione

Per eseguire l’intero workflow:

python src/ames_housing_evaluator.py


Il programma esegue automaticamente:

Analisi del dataset

Preprocessing

Feature selection

Valutazione di regressione, clustering e classificazione

Generazione del riepilogo finale

Possibili estensioni

Hyperparameter tuning (Grid Search / Bayesian Optimization)

Confronto con modelli lineari e boosting

Analisi SHAP per interpretabilità

Versione notebook per presentazione didattica

Pipeline completamente integrata con scikit-learn
