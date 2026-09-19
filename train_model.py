import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'dataset_assurance_ML.csv'
XLSX_PATH = ROOT / 'data' / 'dataset_assurance_ML.xlsx'
MODELS_DIR = ROOT / 'models'
MODEL_PATH = MODELS_DIR / 'pipeline_resiliation.pkl'
METADATA_PATH = MODELS_DIR / 'metadata.json'

TARGET = 'Résiliation'
NUM_COLS = [
    'Âge',
    'Salaire Annuel (€)',
    'Prime Annuelle (€)',
    'Ancienneté (mois)',
    'Coeff. Bonus-Malus',
    'Nb Sinistres (3 ans)',
    'Montant Sinistres (€)',
    'Score Risque (0-100)',
]
CAT_COLS = ['Type Contrat', 'Catégorie Prof.', 'Usage Véhicule', 'Dernier Sinistre']


def train_and_save_model():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    else:
        df = pd.read_excel(XLSX_PATH)
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ('prep', preprocessor),
            ('model', RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
            )),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        'modele': 'Random Forest',
        'auc_test': round(float(roc_auc_score(y_test, y_proba)), 3),
        'num_cols': NUM_COLS,
        'cat_cols': CAT_COLS,
        'num_ranges': {
            col: {
                'min': float(X[col].min()),
                'max': float(X[col].max()),
                'median': float(X[col].median()),
            }
            for col in NUM_COLS
        },
        'cat_values': {
            col: sorted(X[col].dropna().astype(str).unique().tolist())
            for col in CAT_COLS
        },
    }

    with METADATA_PATH.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print('Modèle entraîné avec succès.')
    print(f'Fichier modèle : {MODEL_PATH.exists()}')
    print(f'Fichier metadata : {METADATA_PATH.exists()}')
    print(f'AUC test : {metadata["auc_test"]}')


if __name__ == '__main__':
    train_and_save_model()
