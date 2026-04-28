## Dataset UCI HAR

Télécharger le dataset depuis :
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

```bash
curl -O "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
unzip "UCI HAR Dataset.zip"
```

Structure attendue :
```
data/
└── UCI HAR Dataset/
    ├── train/
    │   ├── X_train.txt
    │   └── y_train.txt
    ├── test/
    │   ├── X_test.txt
    │   └── y_test.txt
    └── activity_labels.txt
```
