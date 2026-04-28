# 🏃 Human Activity Recognition — Deep Learning Comparison

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)
![MLflow](https://img.shields.io/badge/MLflow-2.19-blue?logo=mlflow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Production-green)
![API](https://img.shields.io/badge/API-REST-brightgreen)

> Comparaison complète de 5 architectures Deep Learning (MLP, CNN 1D, LSTM, GRU, CNN+LSTM) pour la reconnaissance automatique d'activités humaines à partir de capteurs smartphone. Tous les modèles sont trackés avec MLflow, versionnés dans le Model Registry et déployés comme API REST.

---

## 📊 Résultats

| Modèle | Accuracy | F1-score | Durée (s) |
|--------|----------|----------|-----------|
| 🥇 **CNN 1D** | **0.9555** | **0.9554** | 174.35 |
| 🥈 **MLP** | **0.9213** | **0.9210** | **15.72** |
| 🥉 CNN+LSTM | 0.8480 | 0.8463 | 615.79 |
| LSTM | 0.8331 | 0.8332 | 1733.58 |
| GRU | 0.8331 | 0.8332 | 1485.88 |

![Comparaison des modèles](results/comparaison_modeles.png)

---

## 🚀 Déploiement Production

### MLflow Model Registry

5 modèles versionnés et enregistrés :

```
HAR_MLP      → version 1
HAR_CNN1D    → version 1 ← déployé en production
HAR_LSTM     → version 1
HAR_GRU      → version 1
HAR_CNN_LSTM → version 1
```

### API REST — MLflow Serve

```bash
export MLFLOW_TRACKING_URI="file:///path/to/mlruns"

mlflow models serve \
  --model-uri "models:/HAR_CNN1D/1" \
  --port 5002 \
  --no-conda \
  --env-manager local
```

### Test de l'API

```python
import requests
import numpy as np
import json

# 561 features capteurs smartphone
data = np.random.randn(1, 561).tolist()

response = requests.post(
    "http://127.0.0.1:5002/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps({"inputs": data})
)

# Résultat : probabilités pour chaque activité
print(response.json())
# {'predictions': [[0.001, 0.0003, 0.0002, 0.9982, 0.000002, 0.0001]]}
#                                              ↑
#                              SITTING avec 99.82% de confiance
```

### Classes prédites

```
0 → WALKING
1 → WALKING_UPSTAIRS
2 → WALKING_DOWNSTAIRS
3 → SITTING
4 → STANDING
5 → LAYING
```

---

## 🎯 Contexte

Identifier automatiquement une activité humaine à partir des capteurs d'un smartphone (accéléromètre + gyroscope). Ce type de système est utilisé dans :
- Les applications de santé et fitness
- Les systèmes de surveillance médicale
- Les interfaces homme-machine intelligentes
- L'IoT et les objets connectés

---

## 📁 Structure du projet

```
projet-dl-har/
├── notebooks/
│   └── projet1_har.ipynb       ← Notebook complet
├── results/
│   └── comparaison_modeles.png ← Graphique comparatif
├── models/                      ← Modèles MLflow
├── data/
│   └── README.md               ← Instructions dataset
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

**UCI HAR Dataset** — University of California, Irvine

| Paramètre | Valeur |
|-----------|--------|
| Échantillons train | 7 352 |
| Échantillons test | 2 947 |
| Features | 561 |
| Classes | 6 |

```bash
cd data
curl -O "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
unzip "UCI HAR Dataset.zip"
```

---

## 🏗️ Architectures

### 🥇 CNN 1D — Meilleure accuracy
```
Input(561,1) → Conv1D(64, k=3, ReLU) → MaxPooling1D(2)
             → Conv1D(128, k=3, ReLU) → MaxPooling1D(2)
             → Flatten → Dense(128, ReLU) → Dropout(0.3)
             → Dense(6, Softmax)
```

### 🥈 MLP — Meilleur ratio vitesse/accuracy
```
Input(561) → Dense(256, ReLU) → Dropout(0.3)
           → Dense(128, ReLU) → Dropout(0.3)
           → Dense(64, ReLU)
           → Dense(6, Softmax)
```

### LSTM
```
Input(561,1) → LSTM(128, return_seq=True) → LSTM(64)
             → Dense(64, ReLU) → Dropout(0.3)
             → Dense(6, Softmax)
```

### GRU
```
Input(561,1) → GRU(128, return_seq=True) → GRU(64)
             → Dense(64, ReLU) → Dropout(0.3)
             → Dense(6, Softmax)
```

### CNN + LSTM
```
Input(561,1) → Conv1D(64, k=3, ReLU) → MaxPooling1D(2)
             → LSTM(128)
             → Dense(64, ReLU) → Dropout(0.3)
             → Dense(6, Softmax)
```

---

## 📈 MLflow Tracking

```python
mlflow.set_experiment("projet1_har")

with mlflow.start_run(run_name="CNN1D"):
    mlflow.log_param("model", "CNN1D")
    mlflow.log_metric("accuracy", 0.9555)
    mlflow.log_metric("f1_score", 0.9554)
    mlflow.tensorflow.log_model(
        model,
        artifact_path="model",
        registered_model_name="HAR_CNN1D"
    )
```

**Lancer MLflow UI :**
```bash
mlflow ui --backend-store-uri file:///path/to/mlruns --port 5001
```

---

## 🔍 Conclusions scientifiques

**1. CNN 1D — Champion (95.55%)**
Les filtres convolutifs 1D détectent des corrélations locales entre features adjacentes, surpassant tous les autres modèles.

**2. MLP — Efficacité maximale**
92.13% d'accuracy en 15 secondes. Sur des features pré-calculées, la simplicité du MLP est un avantage décisif.

**3. LSTM/GRU — Inadaptés aux features pré-calculées**
Conçus pour les signaux bruts, ils sont jusqu'à 110x plus lents que MLP sans gain d'accuracy.

**4. CNN+LSTM — Compromis intéressant**
84.80% — CNN extrait les features avant que LSTM apprenne les patterns temporels.

---

## 🛠️ Installation

```bash
git clone https://github.com/bipanda93/projet-dl-har.git
cd projet-dl-har
conda create -n dl_env python=3.10 -y
conda activate dl_env
pip install -r requirements.txt
cd data
curl -O "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
unzip "UCI HAR Dataset.zip"
cd ..
jupyter notebook notebooks/projet1_har.ipynb
```

---

## 👤 Auteur

**Bipanda Franck Ulrich**
Mastère Data Engineering — Digital School de Paris — Promotion 2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-bipanda--franck-blue?logo=linkedin)](https://linkedin.com/in/franck-bipanda-13392372)
[![Portfolio](https://img.shields.io/badge/Portfolio-datascienceportfol.io-green)](https://datascienceportfol.io/bipandaf)
[![GitHub](https://img.shields.io/badge/GitHub-bipanda93-black?logo=github)](https://github.com/bipanda93)

---

## 📚 Références

- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- Davide Anguita et al. — *A Public Domain Dataset for Human Activity Recognition Using Smartphones* (2013)
- [TensorFlow Documentation](https://www.tensorflow.org)
- [MLflow Documentation](https://mlflow.org)
