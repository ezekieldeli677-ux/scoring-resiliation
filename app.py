import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Scoring Résiliation', page_icon='🚗', layout='wide')

@st.cache_resource
def charger_modele():
	pipeline = joblib.load('models/pipeline_resiliation.pkl')
	with open('models/metadata.json', encoding='utf-8') as f:
		meta = json.load(f)
	return pipeline, meta

pipeline, meta = charger_modele()
num_cols, cat_cols = meta['num_cols'], meta['cat_cols']
rng, cats = meta['num_ranges'], meta['cat_values']

st.sidebar.header('👤 Profil du client')


# petit curseur avec les bornes du metadata
def curseur(col, step=1.0, fmt=None):
	r = rng[col]
	val = st.sidebar.slider(
		col,
		min_value=r['min'],
		max_value=r['max'],
		value=r['median'],
		step=step,
		format=fmt,
	)
	return int(val) if fmt == '%d' else val


# on construit le profil du client
client = {}
client['Âge'] = curseur('Âge', 1.0, '%d')
client['Salaire Annuel (€)'] = curseur('Salaire Annuel (€)', 500.0, '%d')
client['Prime Annuelle (€)'] = curseur('Prime Annuelle (€)', 10.0, '%d')
client['Ancienneté (mois)'] = curseur('Ancienneté (mois)', 1.0, '%d')
client['Coeff. Bonus-Malus'] = curseur('Coeff. Bonus-Malus', 0.01, '%.2f')
client['Nb Sinistres (3 ans)'] = curseur('Nb Sinistres (3 ans)', 1.0, '%d')
client['Montant Sinistres (€)'] = curseur('Montant Sinistres (€)', 100.0, '%d')
client['Score Risque (0-100)'] = curseur('Score Risque (0-100)', 1.0, '%d')

st.sidebar.markdown('---')
# les choix texte du client
for col in cat_cols:
	client[col] = st.sidebar.selectbox(col, cats[col])

st.title('🚗 Scoring de résiliation — Assurance Auto')
st.caption(f"Modèle : {meta['modele']} · AUC test : {meta['auc_test']}")
st.write('Bonjour ! Mon premier modèle en ligne.')

SEUIL_RISQUE = 0.50
SEUIL_MODERE = 0.40

if st.button('🔮 Prédire', type='primary', use_container_width=True):
	# le modele recoit les colonnes dans le bon ordre
	df_client = pd.DataFrame([client])[num_cols + cat_cols]
	proba = float(pipeline.predict_proba(df_client)[0, 1])

	col1, col2 = st.columns([1, 2])
	with col1:
		st.metric('Probabilité de résiliation', f'{proba:.0%}')
		if proba >= SEUIL_RISQUE:
			st.error('⚠️ Client À RISQUE — action de rétention conseillée')
		elif proba >= SEUIL_MODERE:
			st.warning('🟠 Risque modéré — à surveiller')
		else:
			st.success('✅ Client fidèle — risque faible')

	with col2:
		st.write('Niveau de risque')
		st.progress(proba)
		st.write('Données envoyées au modèle :')
		st.dataframe(
			df_client.T.astype(str).rename(columns={0: 'Valeur'}),
			use_container_width=True,
		)

	model = pipeline.named_steps['model']
	if hasattr(model, 'feature_importances_'):
		# on montre les variable les plus importante
		noms = pipeline.named_steps['prep'].get_feature_names_out()
		imp = (
			pd.Series(model.feature_importances_, index=noms)
			.sort_values(ascending=False)
			.head(8)
		)
		imp.index = [n.split('__', 1)[1] for n in imp.index]
		st.subheader('📊 Les 8 variables les plus influentes du modèle')
		st.bar_chart(imp)
else:
	st.info('👈 Ajustez le profil dans la barre latérale, puis cliquez sur Prédire.')

st.subheader('🧪 Tester comme un conseiller')
# quelques profils pour voir si le score parait logique
profil_median = {
	col: rng[col]['median']
	for col in num_cols
}
profil_median.update({
	'Type Contrat': 'Bronze',
	'Catégorie Prof.': 'Employé',
	'Usage Véhicule': 'Domicile-Travail',
	'Dernier Sinistre': 'Aucun',
})

profils_test = {
	'Profil médian': profil_median,
	'Jeune Bronze, 2 sinistres, Vol': {
		**profil_median,
		'Âge': 20,
		'Ancienneté (mois)': 1,
		'Nb Sinistres (3 ans)': 2,
		'Dernier Sinistre': 'Vol',
	},
	'Gold ancien, sans sinistre': {
		**profil_median,
		'Type Contrat': 'Gold',
		'Ancienneté (mois)': 200,
		'Nb Sinistres (3 ans)': 0,
		'Dernier Sinistre': 'Aucun',
	},
}

resultats_test = []
for nom_profil, valeurs in profils_test.items():
	df_profil = pd.DataFrame([valeurs])[num_cols + cat_cols]
	proba_profil = float(pipeline.predict_proba(df_profil)[0, 1])
	resultats_test.append({
		'Profil': nom_profil,
		'Probabilité de résiliation': f'{proba_profil:.1%}',
	})

st.table(pd.DataFrame(resultats_test))
st.caption(
	"Ces résultats sont cohérents avec les corrélations : davantage de sinistres, "
	"un bonus-malus élevé et un profil de risque plus fort augmentent généralement la probabilité. "
	"La ville n'est pas utilisée par le modèle : la modifier seule ne change donc pas la prédiction."
)

primaryColor = "#1F3864"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#EAF3F8"
textColor = "#1B1B1B"