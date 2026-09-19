# Scoring de résiliation 

Application Streamlit qui estime la probabilité de résiliation d'un client à partir de son profil d'assurance.

## Structure du projet

```text
scoring_resiliation/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── dataset_assurance_ML.xlsx
├── models/
│   ├── pipeline_resiliation.pkl
│   └── metadata.json
└── notebooks/
	└── tp_final.ipynb
```

Les dossiers `data/` et `models/` doivent être inclus dans le projet : l'application utilise les données et recharge le pipeline `.pkl` ainsi que ses métadonnées `.json`.

## Lancement

Depuis la racine du projet :

```bash
streamlit run app.py
```

Puis ouvrir http://localhost:8501 dans le navigateur


## Auteur

Projet réalisé par : DELI MHA EZEKIEL  etudiant en informatique à L'ENS de Maroua

lors de la formation de machine learning
