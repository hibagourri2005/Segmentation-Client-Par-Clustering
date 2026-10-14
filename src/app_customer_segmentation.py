# ===================================================================
# APPLICATION DE SEGMENTATION DE CLIENTÈLE AVEC INTERFACE UTILISATEUR
# PROJET PROFESSIONNEL & UNIVERSITAIRE - MACHINE LEARNING (K-MEANS)
# ===================================================================

# =========================
# 1. IMPORT DES LIBRAIRIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="SegmentAI - Segmentation Intelligente de Clientèle",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 2. STYLE ET CSS PERSONNALISÉ
# =========================
def local_css():
    """Injecter du CSS personnalisé"""
    st.markdown("""
    <style>
    /* Style général */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Titres */
    h1 {
        color: #1E3A8A;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 10px;
    }
    
    h2 {
        color: #2563EB;
        border-left: 4px solid #3B82F6;
        padding-left: 10px;
    }
    
    h3 {
        color: #3B82F6;
    }
    
    /* Cartes */
    .custom-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* DataFrames */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    
    /* Badges */
    .cluster-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 2px;
    }
    
    .cluster-0 { background-color: #FF6B6B; color: white; }
    .cluster-1 { background-color: #4ECDC4; color: white; }
    .cluster-2 { background-color: #FFD166; color: black; }
    .cluster-3 { background-color: #06D6A0; color: white; }
    .cluster-4 { background-color: #118AB2; color: white; }
    .cluster-5 { background-color: #7209B7; color: white; }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    </style>
    """, unsafe_allow_html=True)

# =========================
# 3. FONCTIONS UTILITAIRES
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent   # racine du projet
DATA_PATH = BASE_DIR / "data" / "customer_segmentation_mixed.csv"

@st.cache_data
def load_data(file_path=DATA_PATH):
    """Charger les données avec cache et renommer les colonnes si nécessaire"""
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.warning("Fichier non trouvé. Génération de données synthétiques...")
        data = generate_sample_data()
    
    # Renommer les colonnes existantes uniquement
    columns_mapping = {
        'Age': 'Âge',
        'Annual_Income': 'Revenu_Annuel',
        'Spending_Score': 'Score_Dépenses',
        'Monthly_Visits': 'Visites_Mensuelles',
        'Gender': 'Genre',
        'Preferred_Product': 'Produit_Préféré'
    }
    rename_dict = {k: v for k, v in columns_mapping.items() if k in data.columns}
    data.rename(columns=rename_dict, inplace=True)
    
    return data

def generate_sample_data():
    """Générer des données synthétiques pour le projet"""
    np.random.seed(42)
    n_samples = 500
    
    # Données numériques
    age = np.random.randint(18, 70, n_samples)
    annual_income = np.random.normal(60000, 20000, n_samples).clip(20000, 150000)
    spending_score = np.random.randint(1, 100, n_samples)
    monthly_visits = np.random.poisson(4, n_samples).clip(1, 20)
    
    # Données catégorielles
    gender = np.random.choice(['Homme', 'Femme'], n_samples, p=[0.48, 0.52])
    preferred_product = np.random.choice(
        ['Électronique', 'Mode', 'Maison', 'Beauté', 'Sport'], 
        n_samples, 
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    
    data = pd.DataFrame({
        'Âge': age,
        'Revenu_Annuel': annual_income,
        'Score_Dépenses': spending_score,
        'Visites_Mensuelles': monthly_visits,
        'Genre': gender,
        'Produit_Préféré': preferred_product
    })
    
    return data

def preprocess_data(data):
    """Prétraitement des données: nettoyage et encodage"""
    data_clean = data.copy()
    
    # Supprimer les valeurs manquantes
    data_clean.dropna(inplace=True)
    
    # Vérifier que les colonnes existent
    required_cols = ['Genre', 'Produit_Préféré']
    for col in required_cols:
        if col not in data_clean.columns:
            st.warning(f"Colonne manquante: {col}. Encodage impossible.")
            data_clean[col] = 'N/A'  # Valeur par défaut
    
    # Encodage des variables catégorielles
    le_gender = LabelEncoder()
    le_product = LabelEncoder()
    
    data_clean['Genre_encoded'] = le_gender.fit_transform(data_clean['Genre'])
    data_clean['Produit_Préféré_encoded'] = le_product.fit_transform(data_clean['Produit_Préféré'])
    
    return data_clean, le_gender, le_product

def perform_clustering(data, n_clusters=4):
    """Exécuter K-Means sur les colonnes numériques"""
    features = ["Âge", "Revenu_Annuel", "Score_Dépenses", "Visites_Mensuelles"]
    
    # Vérifier que les features existent
    features = [f for f in features if f in data.columns]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Scores d'évaluation
    silhouette = silhouette_score(X_scaled, clusters)
    db_index = davies_bouldin_score(X_scaled, clusters)
    ch_index = calinski_harabasz_score(X_scaled, clusters)
    
    return {
        'scaler': scaler,
        'kmeans': kmeans,
        'clusters': clusters,
        'features_scaled': X_scaled,
        'silhouette_score': silhouette,
        'davies_bouldin_score': db_index,
        'calinski_harabasz_score': ch_index
    }

def predict_new_customer(data, customer_features, scaler, kmeans):
    """Prédire le cluster d'un nouveau client"""
    customer_df = pd.DataFrame([customer_features])
    
    features = ["Âge", "Revenu_Annuel", "Score_Dépenses", "Visites_Mensuelles"]
    features = [f for f in features if f in customer_df.columns]
    
    X_scaled = scaler.transform(customer_df[features])
    cluster = kmeans.predict(X_scaled)[0]
    
    distances = kmeans.transform(X_scaled)[0]
    confidence = 1 - (distances[cluster] / distances.max()) if distances.max() > 0 else 1
    
    return cluster, confidence

def create_cluster_profiles(data, clusters):
    """Créer un profil détaillé pour chaque cluster"""
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters
    
    profiles = {}
    cluster_names = ["🎯 Premium", "🤝 Fidèles", "💰 Économes", "⭐ Standards", 
                     "🚀 Nouveaux", "📈 Croissance", "🎨 Créatifs", "🔧 Techniques"]
    
    for cluster in sorted(data_with_clusters['Cluster'].unique()):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
        cluster_name = cluster_names[cluster % len(cluster_names)]
        
        profile = {
            'Nom': cluster_name,
            'Taille': len(cluster_data),
            'Pourcentage': len(cluster_data) / len(data_with_clusters) * 100,
            'Age_moyen': cluster_data['Âge'].mean() if 'Âge' in cluster_data else np.nan,
            'Revenu_moyen': cluster_data['Revenu_Annuel'].mean() if 'Revenu_Annuel' in cluster_data else np.nan,
            'Score_dépense_moyen': cluster_data['Score_Dépenses'].mean() if 'Score_Dépenses' in cluster_data else np.nan,
            'Visites_moyennes': cluster_data['Visites_Mensuelles'].mean() if 'Visites_Mensuelles' in cluster_data else np.nan,
            'Genre_majoritaire': cluster_data['Genre'].mode()[0] 
                                 if 'Genre' in cluster_data and not cluster_data['Genre'].mode().empty 
                                 else 'N/A',
            'Produit_prefere': cluster_data['Produit_Préféré'].mode()[0] 
                               if 'Produit_Préféré' in cluster_data and not cluster_data['Produit_Préféré'].mode().empty 
                               else 'N/A',
            'Description': '',
            'Couleur': f'cluster-{cluster}'
        }
        
        # Déterminer la description du cluster
        income = profile.get('Revenu_moyen', np.nan)
        spending = profile.get('Score_dépense_moyen', np.nan)
        visits = profile.get('Visites_moyennes', np.nan)
        
        if income > data['Revenu_Annuel'].quantile(0.75) and spending > data['Score_Dépenses'].quantile(0.75):
            profile['Description'] = 'Clients à haute valeur - Fort potentiel de dépenses'
        elif visits > data['Visites_Mensuelles'].quantile(0.75) and spending > data['Score_Dépenses'].quantile(0.5):
            profile['Description'] = 'Clients fidèles - Engagement élevé'
        elif spending < data['Score_Dépenses'].quantile(0.25) and visits < data['Visites_Mensuelles'].quantile(0.5):
            profile['Description'] = 'Clients économes - Potentiel à développer'
        else:
            profile['Description'] = 'Clients standards - Segment stable'
        
        profiles[cluster] = profile
    
    return profiles

def create_gauge_chart(value, title, min_val=0, max_val=1):
    """Créer un graphique en forme de jauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, max_val*0.33], 'color': "lightgray"},
                {'range': [max_val*0.33, max_val*0.66], 'color': "gray"},
                {'range': [max_val*0.66, max_val], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def get_marketing_recommendations(description):
    """Retourner des recommandations marketing basées sur la description du segment"""
    recommendations = {
        "haute valeur": """
        **Stratégie Premium:**
        1. **Offres exclusives**: Produits/services haut de gamme
        2. **Service personnalisé**: Consultant dédié, support prioritaire
        3. **Avantages VIP**: Accès anticipé, événements privés
        4. **Programme fidélité élite**: Récompenses exclusives
        """,
        "fidèles": """
        **Stratégie Relationnelle:**
        1. **Programme fidélité**: Points cumulables, avantages croissants
        2. **Communication régulière**: Newsletters personnalisées, offres anniversaire
        3. **Engagement communautaire**: Groupes clients, événements
        4. **Récompenses**: Cashback, cadeaux exclusifs
        """,
        "économes": """
        **Stratégie Incitative:**
        1. **Promotions ciblées**: Coupons, réductions, soldes
        2. **Échantillons gratuits**: Produits découverte
        3. **Éducation client**: Guides, témoignages, comparaisons
        4. **Bundles économiques**: Packs à prix avantageux
        """,
        "standards": """
        **Stratégie Équilibrée:**
        1. **Marketing général**: Campagnes segmentées basiques
        2. **Offres saisonnières**: Promotions périodiques
        3. **Email marketing**: Communications régulières
        4. **Upselling**: Suggestions basées sur l'historique
        """
    }
    
    for key, value in recommendations.items():
        if key in description.lower():
            return value
    
    return """
    **Stratégie Générale:**
    1. Analyse continue du comportement
    2. Tests A/B pour optimiser les offres
    3. Collecte de feedback régulière
    4. Adaptation progressive des stratégies
    """

def show_cluster_details(profile, cluster_id, data):
    """Afficher les détails complets d'un cluster"""
    # Filtrer les données du cluster
    cluster_data = data[data['Cluster'] == cluster_id]
    
    # Métriques détaillées
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Caractéristiques Principales:**")
        st.write(f"- Genre majoritaire: **{profile.get('Genre_majoritaire', 'N/A')}**")
        st.write(f"- Produit préféré: **{profile.get('Produit_prefere', 'N/A')}**")
        
        # Vérifier si les colonnes existent
        if 'Âge' in cluster_data.columns:
            st.write(f"- Âge médian: **{cluster_data['Âge'].median():.1f}** ans")
        else:
            st.write(f"- Âge médian: **N/A**")
            
        if 'Revenu_Annuel' in cluster_data.columns:
            st.write(f"- Revenu médian: **${cluster_data['Revenu_Annuel'].median():,.0f}**")
        else:
            st.write(f"- Revenu médian: **N/A**")
    
    with col2:
        st.markdown("**📊 Distribution:**")
        
        if 'Revenu_Annuel' in cluster_data.columns:
            st.write(f"- Écart-type revenu: **${cluster_data['Revenu_Annuel'].std():,.0f}**")
        else:
            st.write(f"- Écart-type revenu: **N/A**")
            
        if 'Score_Dépenses' in cluster_data.columns:
            st.write(f"- Min score dépense: **{cluster_data['Score_Dépenses'].min():.0f}**")
            st.write(f"- Max score dépense: **{cluster_data['Score_Dépenses'].max():.0f}**")
        else:
            st.write(f"- Score dépense: **N/A**")
            
        if 'Genre' in cluster_data.columns:
            hommes = (cluster_data['Genre'] == 'Homme').sum()
            femmes = (cluster_data['Genre'] == 'Femme').sum()
            if femmes > 0:
                ratio = hommes / femmes
                st.write(f"- Ratio hommes/femmes: **{ratio:.2f}**")
            else:
                st.write(f"- Ratio hommes/femmes: **N/A**")
        else:
            st.write(f"- Genre: **N/A**")
    
    # Visualisations spécifiques au cluster
    st.markdown("**📊 Analyse du Segment:**")
    
    tab1, tab2, tab3 = st.tabs(["Genre", "Produits", "Distribution"])
    
    with tab1:
        if 'Genre' in cluster_data.columns:
            fig = px.pie(cluster_data, names='Genre', 
                        title=f"Répartition par Genre - {profile.get('Nom', f'Segment {cluster_id}')}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données de genre non disponibles")
    
    with tab2:
        if 'Produit_Préféré' in cluster_data.columns:
            product_counts = cluster_data['Produit_Préféré'].value_counts()
            fig = px.bar(x=product_counts.index, y=product_counts.values,
                        title=f"Produits Préférés - {profile.get('Nom', f'Segment {cluster_id}')}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données de produits préférés non disponibles")
    
    with tab3:
        if 'Revenu_Annuel' in cluster_data.columns:
            fig = px.histogram(cluster_data, x='Revenu_Annuel', nbins=20,
                              title=f"Distribution des Revenus - {profile.get('Nom', f'Segment {cluster_id}')}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données de revenu non disponibles")
    
    # Top 5 clients représentatifs
    st.markdown("**👥 Clients Représentatifs:**")
    
    if len(cluster_data) > 0:
        representative_clients = cluster_data.head(5)
        
        for idx, (_, client) in enumerate(representative_clients.iterrows(), 1):
            age = client.get('Âge', 'N/A')
            revenu = client.get('Revenu_Annuel', 'N/A')
            score = client.get('Score_Dépenses', 'N/A')
            visites = client.get('Visites_Mensuelles', 'N/A')
            
            age_str = f"{int(age)}" if isinstance(age, (int, float)) else str(age)
            revenu_str = f"${int(revenu):,}" if isinstance(revenu, (int, float)) else str(revenu)
            score_str = f"{int(score)}" if isinstance(score, (int, float)) else str(score)
            visites_str = f"{int(visites)}" if isinstance(visites, (int, float)) else str(visites)
            
            st.write(f"- **Client #{idx}**: {age_str} ans, {revenu_str} revenu, Score: {score_str}, "
                    f"Visites: {visites_str}/mois")
    else:
        st.info("Aucun client disponible dans ce segment")

# =========================
# 4. FONCTIONS D'AFFICHAGE PRINCIPALES
# =========================

def show_dashboard(data):
    """Afficher le tableau de bord principal"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; color: white; margin-bottom: 30px;">
        <h1 style="color: white; margin-bottom: 10px;">Tableau de Bord SegmentAI</h1>
        <p style="font-size: 1.2rem;">Analyse intelligente de votre clientèle avec Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>👥 Clients Totaux</h3>
            <h2>{:,}</h2>
        </div>
        """.format(len(data)), unsafe_allow_html=True)
    
    with col2:
        # Gérer différentes dénominations de colonnes
        revenu_col = None
        if 'Revenu_Annuel' in data.columns:
            revenu_col = 'Revenu_Annuel'
        elif 'Annual_Income' in data.columns:
            revenu_col = 'Annual_Income'
        
        avg_income = data[revenu_col].mean() if revenu_col else 0
        st.markdown("""
        <div class="metric-card">
            <h3>💰 Revenu Moyen</h3>
            <h2>${:,.0f}</h2>
        </div>
        """.format(avg_income), unsafe_allow_html=True)
    
    with col3:
        # Gérer différentes dénominations de colonnes
        age_col = None
        if 'Âge' in data.columns:
            age_col = 'Âge'
        elif 'Age' in data.columns:
            age_col = 'Age'
        
        avg_age = data[age_col].mean() if age_col else 0
        st.markdown("""
        <div class="metric-card">
            <h3>📅 Âge Moyen</h3>
            <h2>{:.0f} ans</h2>
        </div>
        """.format(avg_age), unsafe_allow_html=True)
    
    with col4:
        # Gérer différentes dénominations de colonnes
        product_col = None
        if 'Produit_Préféré' in data.columns:
            product_col = 'Produit_Préféré'
        elif 'Preferred_Product' in data.columns:
            product_col = 'Preferred_Product'
        
        unique_products = data[product_col].nunique() if product_col else 0
        st.markdown("""
        <div class="metric-card">
            <h3>🏷️ Catégories</h3>
            <h2>{}</h2>
        </div>
        """.format(unique_products), unsafe_allow_html=True)
    
    # Aperçu rapide des données
    st.markdown("### 📈 Aperçu des Données")
    
    tab1, tab2, tab3 = st.tabs(["📊 Données Brutes", "📈 Statistiques", "🔍 Prévisualisation"])
    
    with tab1:
        st.dataframe(data.head(10), use_container_width=True)
    
    with tab2:
        st.dataframe(data.describe().style.format("{:.2f}"), use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            # Distribution de l'âge
            if age_col:
                fig = px.histogram(data, x=age_col, nbins=30, title="Distribution de l'Âge")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Répartition par genre
            genre_col = None
            if 'Genre' in data.columns:
                genre_col = 'Genre'
            elif 'Gender' in data.columns:
                genre_col = 'Gender'
            
            if genre_col:
                fig = px.pie(data, names=genre_col, title="Répartition par Genre")
                st.plotly_chart(fig, use_container_width=True)
    
    # Recommandations rapides
    st.markdown("### 🚀 Actions Recommandées")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("""
            <div style="background: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3;">
                <h4>🎯 Segmentation Initiale</h4>
                <p>Exécutez une première analyse avec 4 segments pour découvrir vos profils clients.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown("""
            <div style="background: #F3E5F5; padding: 20px; border-radius: 10px; border-left: 5px solid #9C27B0;">
                <h4>📊 Analyse des Tendances</h4>
                <p>Explorez les relations entre revenus et comportements d'achat.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown("""
            <div style="background: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                <h4>🔮 Prédiction Client</h4>
                <p>Testez notre outil de prédiction pour classer de nouveaux clients.</p>
            </div>
            """, unsafe_allow_html=True)

def show_analysis(data):
    """Afficher l'analyse exploratoire des données"""
    st.header("📊 Analyse Exploratoire Avancée")
    
    # Onglets pour différentes analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Corrélations", "📊 Comparaisons", "📋 Insights"])
    
    with tab1:
        st.subheader("Distribution des Variables Numériques")
        
        # Déterminer les noms de colonnes
        num_features = []
        if 'Âge' in data.columns and 'Revenu_Annuel' in data.columns:
            num_features = ["Âge", "Revenu_Annuel", "Score_Dépenses", "Visites_Mensuelles"]
        elif 'Age' in data.columns:
            num_features = ["Age", "Annual_Income", "Spending_Score", "Monthly_Visits"]
        
        for i in range(0, len(num_features), 2):
            cols = st.columns(2)
            for j, feature in enumerate(num_features[i:i+2]):
                if j < len(cols) and feature in data.columns:
                    with cols[j]:
                        fig = px.histogram(data, x=feature, nbins=30, 
                                         marginal="box", 
                                         title=f"Distribution de {feature}")
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Matrice de Corrélation")
        
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            corr_matrix = numeric_data.corr()
            fig = px.imshow(corr_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          color_continuous_scale='RdBu_r',
                          title="Matrice de Corrélation")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top corrélations
            st.subheader("Top des Corrélations")
            corr_pairs = corr_matrix.unstack()
            sorted_pairs = corr_pairs.sort_values(ascending=False)
            top_pairs = sorted_pairs[sorted_pairs != 1].head(10)
            
            for (var1, var2), value in top_pairs.items():
                st.metric(f"{var1} ↔ {var2}", f"{value:.3f}")
    
    with tab3:
        st.subheader("Comparaisons par Catégorie")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Genre' in data.columns or 'Gender' in data.columns:
                genre_col = 'Genre' if 'Genre' in data.columns else 'Gender'
                if 'Revenu_Annuel' in data.columns:
                    fig = px.box(data, x=genre_col, y='Revenu_Annuel', title="Revenu par Genre")
                elif 'Annual_Income' in data.columns:
                    fig = px.box(data, x=genre_col, y='Annual_Income', title="Revenu par Genre")
                if 'fig' in locals():
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Produit_Préféré' in data.columns or 'Preferred_Product' in data.columns:
                product_col = 'Produit_Préféré' if 'Produit_Préféré' in data.columns else 'Preferred_Product'
                if 'Score_Dépenses' in data.columns:
                    fig = px.box(data, x=product_col, y='Score_Dépenses', title="Score de Dépenses par Produit")
                elif 'Spending_Score' in data.columns:
                    fig = px.box(data, x=product_col, y='Spending_Score', title="Score de Dépenses par Produit")
                if 'fig' in locals():
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Insights Automatiques")
        
        # Calculer des insights
        insights = []
        
        if 'Revenu_Annuel' in data.columns:
            skewness = data['Revenu_Annuel'].skew()
            if skewness > 0.5:
                insights.append("📊 **Distribution asymétrique** : Les revenus sont concentrés vers les valeurs basses")
            elif skewness < -0.5:
                insights.append("📊 **Distribution asymétrique** : Les revenus sont concentrés vers les valeurs hautes")
        elif 'Annual_Income' in data.columns:
            skewness = data['Annual_Income'].skew()
            if skewness > 0.5:
                insights.append("📊 **Distribution asymétrique** : Les revenus sont concentrés vers les valeurs basses")
            elif skewness < -0.5:
                insights.append("📊 **Distribution asymétrique** : Les revenus sont concentrés vers les valeurs hautes")
        
        if 'Score_Dépenses' in data.columns:
            if data['Score_Dépenses'].std() > 20:
                insights.append("🎯 **Grande variabilité** : Les scores de dépenses sont très diversifiés")
        elif 'Spending_Score' in data.columns:
            if data['Spending_Score'].std() > 20:
                insights.append("🎯 **Grande variabilité** : Les scores de dépenses sont très diversifiés")
        
        if 'Visites_Mensuelles' in data.columns:
            avg_visits = data['Visites_Mensuelles'].mean()
            if avg_visits < 5:
                insights.append("⚠️ **Engagement modéré** : Fréquence de visites moyenne à améliorer")
        elif 'Monthly_Visits' in data.columns:
            avg_visits = data['Monthly_Visits'].mean()
            if avg_visits < 5:
                insights.append("⚠️ **Engagement modéré** : Fréquence de visites moyenne à améliorer")
        
        # Afficher les insights
        for insight in insights:
            st.markdown(f"""
            <div style="background: #FFF3E0; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #FF9800;">
                {insight}
            </div>
            """, unsafe_allow_html=True)

def show_clustering(data, n_clusters):
    """Afficher les résultats du clustering"""
    st.header("🎯 Segmentation Intelligente par K-Means")
    
    with st.spinner("🔍 Analyse des données en cours..."):
        # Prétraitement
        data_clean, _, _ = preprocess_data(data)
        
        # Exécuter le clustering
        results = perform_clustering(data_clean, n_clusters)
        data_clean['Cluster'] = results['clusters']
    
    # Métriques de performance
    st.subheader("📊 Performance du Modèle")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = create_gauge_chart(results['silhouette_score'], "Score Silhouette")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        db_score_norm = 1 - min(results['davies_bouldin_score'] / 2, 1)
        fig = create_gauge_chart(db_score_norm, "Davies-Bouldin")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.metric("Inertie", f"{results['kmeans'].inertia_:,.0f}", delta="- Plus bas est meilleur")
    
    with col4:
        st.metric("Segments", n_clusters)
    
    # Visualisation des clusters
    st.markdown("### 🎨 Visualisation des Segments")
    
    tab1, tab2, tab3 = st.tabs(["2D - PCA", "3D - Analyse", "📊 Répartition"])
    
    with tab1:
        # Réduction PCA 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(results['features_scaled'])
        
        fig = px.scatter(
            x=X_pca[:, 0], 
            y=X_pca[:, 1],
            color=data_clean['Cluster'].astype(str),
            hover_data={
                'Client': range(len(data_clean)),
                'Revenu': data_clean['Revenu_Annuel'] if 'Revenu_Annuel' in data_clean.columns else None,
                'Âge': data_clean['Âge'] if 'Âge' in data_clean.columns else None
            },
            labels={'color': 'Segment'},
            title="Projection 2D des Segments (PCA)",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Analyse 3D si plus de 2 composantes
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(results['features_scaled'])
        
        fig = px.scatter_3d(
            x=X_pca_3d[:, 0], 
            y=X_pca_3d[:, 1],
            z=X_pca_3d[:, 2],
            color=data_clean['Cluster'].astype(str),
            title="Visualisation 3D des Segments"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Distribution des clusters
        cluster_dist = data_clean['Cluster'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=cluster_dist.values, 
                        names=[f"Segment {i}" for i in cluster_dist.index],
                        title="Répartition des Segments")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=cluster_dist.index, y=cluster_dist.values,
                        title="Taille des Segments",
                        labels={'x': 'Segment', 'y': 'Nombre de Clients'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Graphique revenu vs dépenses
    st.markdown("### 💰 Analyse Revenu vs Dépenses par Segment")
    
    if 'Revenu_Annuel' in data_clean.columns and 'Score_Dépenses' in data_clean.columns:
        fig = px.scatter(
            data_clean,
            x="Revenu_Annuel",
            y="Score_Dépenses",
            color="Cluster",
            size="Visites_Mensuelles" if 'Visites_Mensuelles' in data_clean.columns else None,
            hover_data=["Âge", "Genre", "Produit_Préféré"],
            title="Segmentation: Revenu vs Comportement de Dépense",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Optimisation du nombre de clusters
    st.markdown("### 📈 Optimisation du Nombre de Segments")
    
    with st.expander("🔍 Analyse de l'Optimalité", expanded=True):
        # Calcul des métriques pour différents k
        wcss = []
        silhouette_scores = []
        db_scores = []
        k_range = range(2, 11)
        
        progress_bar = st.progress(0)
        for idx, k in enumerate(k_range):
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters_temp = kmeans_temp.fit_predict(results['features_scaled'])
            wcss.append(kmeans_temp.inertia_)
            silhouette_scores.append(silhouette_score(results['features_scaled'], clusters_temp))
            db_scores.append(davies_bouldin_score(results['features_scaled'], clusters_temp))
            progress_bar.progress((idx + 1) / len(k_range))
        
        # Graphiques
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=wcss, name='Inertie', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, name='Silhouette', 
                                yaxis='y2', line=dict(color='green')))
        
        fig.update_layout(
            title="Métriques d'Optimisation",
            xaxis_title="Nombre de Segments",
            yaxis=dict(title="Inertie (WCSS)", color="blue"),
            yaxis2=dict(title="Score Silhouette", color="green", overlaying="y", side="right"),
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommandation
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        st.success(f"✅ **Recommandation** : {optimal_k} segments optimisent le score Silhouette ({max(silhouette_scores):.3f})")

def show_prediction(data):
    """Interface de prédiction pour un nouveau client"""
    st.header("🔮 Prédiction de Segment Client")
    
    st.markdown("""
    <div style="background: #E8F5E9; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="color: #2E7D32;">💡 Comment ça marche ?</h4>
        <p>Entrez les informations d'un nouveau client pour prédire à quel segment il appartient. 
        Notre modèle K-Means classera automatiquement le client et fournira des recommandations personnalisées.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Formulaire en deux colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Informations Démographiques")
        age = st.slider("Âge", 18, 80, 35, help="Âge du client")
        
        gender_options = ["Homme", "Femme", "Autre"]
        gender = st.selectbox("Genre", gender_options)
        
        income = st.number_input("Revenu Annuel (€)", 20000, 200000, 50000, step=1000,
                               help="Revenu annuel brut estimé")
    
    with col2:
        st.markdown("### 🛍️ Comportement d'Achat")
        spending_score = st.slider("Score de Dépense", 1, 100, 50, 
                                 help="Tendance à dépenser (1=économe, 100=dépensier)")
        
        monthly_visits = st.slider("Visites Mensuelles", 1, 30, 4,
                                 help="Nombre moyen de visites par mois")
        
        product_options = ["Électronique", "Mode", "Maison", "Beauté", "Sport", "Alimentation", "Loisirs"]
        preferred_product = st.selectbox("Produit Préféré", product_options)
    
    # Bouton de prédiction stylé
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🎯 Prédire le Segment", 
                                  type="primary", 
                                  use_container_width=True,
                                  help="Cliquez pour exécuter la prédiction")
    
    if predict_button:
        with st.spinner("🔍 Analyse du profil client en cours..."):
            try:
                # Préparation des données
                data_clean, _, _ = preprocess_data(data)
                results = perform_clustering(data_clean, 4)
                
                # Création du profil client
                customer_features = {
                    "Âge": age,
                    "Revenu_Annuel": income,
                    "Score_Dépenses": spending_score,
                    "Visites_Mensuelles": monthly_visits,
                    "Genre": gender,
                    "Produit_Préféré": preferred_product
                }
                
                # Prédiction
                cluster, confidence = predict_new_customer(data_clean, customer_features, 
                                                          results['scaler'], results['kmeans'])
                
                # Affichage des résultats
                st.markdown("---")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 30px; border-radius: 15px; color: white; text-align: center;">
                    <h2 style="color: white;">Résultat de la Prédiction</h2>
                    <div class="cluster-badge cluster-{cluster}" style="font-size: 1.5rem; margin: 20px auto;">
                        Segment {cluster}
                    </div>
                    <p style="font-size: 1.2rem;">Confiance: <strong>{confidence*100:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Détails du segment
                data_clean['Cluster'] = results['clusters']
                profiles = create_cluster_profiles(data_clean, results['clusters'])
                
                if cluster in profiles:
                    profile = profiles[cluster]
                    
                    st.markdown(f"### 📊 Caractéristiques du Segment {cluster}")
                    
                    # Vérifier que toutes les clés existent avant d'accéder
                    taille = profile.get('Taille', 'N/A')
                    pourcentage = profile.get('Pourcentage', 0)
                    revenu_moyen = profile.get('Revenu_moyen', 0)
                    score_depense = profile.get('Score_dépense_moyen', 0)
                    
                    # Métriques du segment
                    cols = st.columns(4)
                    metrics = [
                        ("👥 Taille", f"{taille} clients"),
                        ("📈 Part", f"{pourcentage:.1f}%"),
                        ("💰 Revenu moyen", f"${revenu_moyen:,.0f}" if isinstance(revenu_moyen, (int, float)) else str(revenu_moyen)),
                        ("🎯 Score dépense", f"{score_depense:.1f}/100" if isinstance(score_depense, (int, float)) else str(score_depense))
                    ]
                    
                    for col, (label, value) in zip(cols, metrics):
                        with col:
                            st.metric(label, value)
                    
                    # Tableau détaillé
                    st.markdown("#### 📋 Profil Détailé")
                    detail_cols = st.columns(2)
                    
                    with detail_cols[0]:
                        age_moyen = profile.get('Age_moyen', 'N/A')
                        genre_majoritaire = profile.get('Genre_majoritaire', 'N/A')
                        visites_moyennes = profile.get('Visites_moyennes', 'N/A')
                        
                        age_str = f"{age_moyen:.1f} ans" if isinstance(age_moyen, (int, float)) else str(age_moyen)
                        visites_str = f"{visites_moyennes:.1f}" if isinstance(visites_moyennes, (int, float)) else str(visites_moyennes)
                        
                        st.markdown(f"""
                        **Démographie:**
                        - Âge moyen: {age_str}
                        - Genre majoritaire: {genre_majoritaire}
                        - Visites/mois: {visites_str}
                        """)
                    
                    with detail_cols[1]:
                        produit_prefere = profile.get('Produit_prefere', 'N/A')
                        description = profile.get('Description', 'N/A')
                        nom = profile.get('Nom', f'Segment {cluster}')
                        
                        st.markdown(f"""
                        **Comportement:**
                        - Produit préféré: {produit_prefere}
                        - Description: {description}
                        - Nom du segment: {nom}
                        """)
                    
                    # Recommandations marketing
                    st.markdown("### 💡 Stratégies Marketing Recommandées")
                    
                    recommendation_colors = {
                        "Premium": "#4CAF50",
                        "Fidèles": "#2196F3",
                        "Économes": "#FF9800",
                        "Standards": "#9C27B0"
                    }
                    
                    # Extraire le type de segment du nom
                    segment_type = nom.split()[1] if len(nom.split()) > 1 else "Standards"
                    rec_color = recommendation_colors.get(segment_type, "#757575")
                    
                    st.markdown(f"""
                    <div style="background: {rec_color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {rec_color};">
                        <h4 style="color: {rec_color};">🎯 Approche recommandée pour le segment</h4>
                        {get_marketing_recommendations(description)}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualisation
                    st.markdown("### 📍 Position du Client dans l'Espace des Segments")
                    
                    try:
                        # Ajouter le nouveau client aux données pour visualisation
                        viz_data = data_clean.copy()
                        viz_data['Type'] = 'Client existant'
                        viz_data['Taille'] = 5
                        
                        new_client_row = pd.DataFrame([{
                            'Âge': age,
                            'Revenu_Annuel': income,
                            'Score_Dépenses': spending_score,
                            'Visites_Mensuelles': monthly_visits,
                            'Genre': gender,
                            'Produit_Préféré': preferred_product,
                            'Cluster': cluster,
                            'Type': 'Nouveau client',
                            'Taille': 15
                        }])
                        
                        viz_data = pd.concat([viz_data, new_client_row], ignore_index=True)
                        
                        fig = px.scatter(
                            viz_data,
                            x="Revenu_Annuel",
                            y="Score_Dépenses",
                            color="Cluster",
                            symbol="Type",
                            size="Taille",
                            hover_data=["Âge", "Visites_Mensuelles", "Genre"],
                            title="Position relative du nouveau client",
                            color_discrete_sequence=px.colors.qualitative.Set1
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"Impossible de générer la visualisation: {e}")
                        st.info("Les données peuvent manquer de certaines colonnes nécessaires.")
                
                else:
                    st.warning(f"Aucun profil trouvé pour le segment {cluster}")
                    st.info("Le segment prédit n'existe pas dans les profils générés.")
                    
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {str(e)}")
                st.info("Veuillez vérifier que les données d'entrée sont correctes.")
    
    # Section d'aide
    with st.expander("❓ Aide et informations supplémentaires"):
        st.markdown("""
        **Comment interpréter les résultats:**
        
        1. **Segment**: Le groupe de clients auquel appartient le nouveau client
        2. **Confiance**: La fiabilité de la prédiction (plus élevé = plus fiable)
        3. **Score de dépense**: 1-100 (économe à dépensier)
        4. **Revenu moyen**: Revenu annuel moyen du segment
        
        **Conseils:**
        - Pour des résultats optimaux, utilisez des données réelles
        - Le modèle s'améliore avec plus de données
        - Les recommandations sont basées sur le comportement moyen du segment
        """)

def show_profiles(data, n_clusters):
    """Afficher les profils détaillés des clusters"""
    st.header("📋 Profils Détailés des Segments Clients")
    
    # Générer les profils
    with st.spinner("🔍 Création des profils en cours..."):
        data_clean, _, _ = preprocess_data(data)
        results = perform_clustering(data_clean, n_clusters)
        data_clean['Cluster'] = results['clusters']
        profiles = create_cluster_profiles(data_clean, results['clusters'])
    
    # Vue d'ensemble
    st.markdown("### 📊 Vue d'Ensemble des Segments")
    
    # Créer une grille de cartes pour chaque segment
    for i in range(0, len(profiles), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(profiles):
                cluster = list(profiles.keys())[i + j]
                profile = profiles[cluster]
                
                with cols[j]:
                    # Carte du segment
                    st.markdown(f"""
                    <div style="background: white; border-radius: 10px; padding: 20px; 
                                box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #1E3A8A;">{profile['Nom']}</h3>
                            <span class="cluster-badge {profile['Couleur']}">Segment {cluster}</span>
                        </div>
                        <p style="color: #6B7280; margin: 10px 0;">{profile['Description']}</p>
                        <div style="background: #F3F4F6; padding: 15px; border-radius: 8px; margin: 10px 0;">
                            <p style="margin: 5px 0;">👥 <strong>{profile['Taille']} clients</strong> ({profile['Pourcentage']:.1f}%)</p>
                            <p style="margin: 5px 0;">💰 <strong>${profile['Revenu_moyen']:,.0f}</strong> revenu moyen</p>
                            <p style="margin: 5px 0;">🎯 <strong>{profile['Score_dépense_moyen']:.1f}</strong> score dépense</p>
                            <p style="margin: 5px 0;">📅 <strong>{profile['Visites_moyennes']:.1f}</strong> visites/mois</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Bouton pour voir les détails
                    with st.expander("📈 Voir les détails complets"):
                        show_cluster_details(profile, cluster, data_clean)
    
    # Tableau comparatif
    st.markdown("---")
    st.markdown("### 📊 Tableau Comparatif")
    
    # Créer le dataframe comparatif
    comparative_data = []
    for cluster, profile in profiles.items():
        comparative_data.append({
            'Segment': f"Segment {cluster}",
            'Nom': profile['Nom'],
            'Clients': profile['Taille'],
            'Part (%)': profile['Pourcentage'],
            'Âge Moyen': profile['Age_moyen'],
            'Revenu Moyen': profile['Revenu_moyen'],
            'Score Dépense': profile['Score_dépense_moyen'],
            'Visites/Mois': profile['Visites_moyennes']
        })
    
    comparative_df = pd.DataFrame(comparative_data)
    
    # Afficher avec style
    st.dataframe(
        comparative_df.style
        .background_gradient(subset=['Part (%)', 'Revenu Moyen'], cmap='Blues')
        .format({
            'Part (%)': '{:.1f}%',
            'Âge Moyen': '{:.1f}',
            'Revenu Moyen': '${:,.0f}',
            'Score Dépense': '{:.1f}',
            'Visites/Mois': '{:.1f}'
        })
        .set_properties(**{'text-align': 'center'}),
        use_container_width=True,
        height=300
    )
    
    # Téléchargement des profils
    st.markdown("### 💾 Export des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Télécharger les profils (CSV)", use_container_width=True):
            csv = comparative_df.to_csv(index=False)
            st.download_button(
                label="Cliquez pour télécharger",
                data=csv,
                file_name="profils_segments.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Générer un rapport complet", use_container_width=True):
            st.info("Fonctionnalité en développement...")

def show_settings():
    """Afficher les paramètres de l'application"""
    st.header("⚙️ Paramètres de l'Application")
    
    st.markdown("### 🔧 Configuration du Modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        algorithm = st.selectbox(
            "Algorithme de clustering",
            ["K-Means", "DBSCAN", "Agglomerative", "Gaussian Mixture"],
            help="Sélectionnez l'algorithme de clustering à utiliser"
        )
        
        max_iterations = st.slider(
            "Nombre maximum d'itérations",
            min_value=100,
            max_value=1000,
            value=300,
            step=50
        )
    
    with col2:
        random_state = st.number_input(
            "Seed aléatoire",
            min_value=0,
            max_value=1000,
            value=42,
            help="Pour la reproductibilité des résultats"
        )
        
        n_init = st.slider(
            "Nombre d'initialisations",
            min_value=1,
            max_value=20,
            value=10,
            help="Nombre de fois où K-Means sera exécuté avec différents centroïdes initiaux"
        )
    
    st.markdown("### 🎨 Personnalisation de l'Interface")
    
    theme = st.selectbox(
        "Thème de l'interface",
        ["Clair", "Sombre", "Automatique"]
    )
    
    color_scheme = st.color_picker(
        "Couleur principale",
        value="#3B82F6"
    )
    
    st.markdown("### 💾 Gestion des Données")
    
    if st.button("🗑️ Effacer le cache", help="Efface toutes les données mises en cache"):
        st.cache_data.clear()
        st.success("Cache effacé avec succès!")
    
    if st.button("🔄 Réinitialiser les paramètres"):
        st.success("Paramètres réinitialisés aux valeurs par défaut")
        st.rerun()

# =========================
# 5. FONCTION PRINCIPALE
# =========================
def main():
    """Fonction principale de l'application"""
    # Appliquer le CSS personnalisé
    local_css()
    
    # Sidebar avec design amélioré
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #1E3A8A; margin-bottom: 0;">🤖 SegmentAI</h1>
            <p style="color: #6B7280; font-size: 0.9rem;">Intelligence Artificielle de Segmentation Clientèle</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Menu de navigation stylé
        selected = option_menu(
            menu_title="Navigation",
            options=["🏠 Tableau de Bord", "📊 Analyse", "🎯 Segmentation", "🔮 Prédiction", "📋 Profils", "⚙️ Paramètres"],
            icons=["house", "graph-up", "bullseye", "magic", "person-lines-fill", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#f8f9fa"},
                "icon": {"color": "#3B82F6", "font-size": "18px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#e2e8f0",
                },
                "nav-link-selected": {
                    "background-color": "#3B82F6",
                    "color": "white",
                },
            }
        )
        
        st.markdown("---")
        
        # Upload de fichier
        st.markdown("### 📁 Chargement des données")
        uploaded_file = st.file_uploader(
            "Importer un fichier CSV",
            type=['csv'],
            help="Format attendu: Âge, Revenu_Annuel, Score_Dépenses, Visites_Mensuelles, Genre, Produit_Préféré"
        )
        
        # Paramètres du modèle
        st.markdown("### ⚙️ Configuration")
        n_clusters = st.slider(
            "Nombre de segments",
            min_value=2,
            max_value=8,
            value=4,
            help="Sélectionnez le nombre optimal de segments clients"
        )
        
        # Bouton de rafraîchissement
        if st.button("🔄 Rafraîchir l'analyse", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # Footer de la sidebar
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
            <p>SegmentAI v1.0</p>
            <p>© 2024 - Projet Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Chargement des données
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = load_data(DATA_PATH)
    
    # Contenu principal basé sur la sélection
    if selected == "🏠 Tableau de Bord":
        show_dashboard(data)
    elif selected == "📊 Analyse":
        show_analysis(data)
    elif selected == "🎯 Segmentation":
        show_clustering(data, n_clusters)
    elif selected == "🔮 Prédiction":
        show_prediction(data)
    elif selected == "📋 Profils":
        show_profiles(data, n_clusters)
    elif selected == "⚙️ Paramètres":
        show_settings()

# =========================
# 6. EXÉCUTION PRINCIPALE
# =========================
if __name__ == "__main__":
    main()