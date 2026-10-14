# ===================================================================
# SEGMENTATION DE CLIENTELE PAR MACHINE LEARNING (K-MEANS)
# PROJET PROFESSIONNEL & UNIVERSITAIRE - SCRIPT PYTHON
# ===================================================================

# =========================
# 1. IMPORT DES LIBRAIRIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =========================
# 2. CONFIGURATION DES CHEMINS
# =========================
# Créer ou utiliser le dossier resultats
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent   # racine du projet
DATA_PATH = BASE_DIR / "data" / "customer_segmentation_mixed.csv"
RESULTATS_DIR = str(BASE_DIR / "resultats")
os.makedirs(RESULTATS_DIR, exist_ok=True)

# =========================
# 3. FONCTIONS UTILITAIRES
# =========================
def charger_donnees():
    """Charger les donnees"""
    try:
        data = pd.read_csv(DATA_PATH)
        print(f"Donnees chargees avec succes depuis '{DATA_PATH}'")
        return data
    except FileNotFoundError:
        print("Fichier non trouve. Generation de donnees synthetiques...")
        return generer_donnees_echantillon()

def generer_donnees_echantillon():
    """Generer des donnees synthetiques pour le projet"""
    np.random.seed(42)
    n_samples = 500
    
    # Donnees numeriques
    age = np.random.randint(18, 70, n_samples)
    annual_income = np.random.normal(60000, 20000, n_samples).clip(20000, 150000)
    spending_score = np.random.randint(1, 100, n_samples)
    monthly_visits = np.random.poisson(4, n_samples).clip(1, 20)
    
    # Donnees categoriques
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
    preferred_product = np.random.choice(
        ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports'], 
        n_samples, 
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    
    data = pd.DataFrame({
        'Age': age,
        'Annual_Income': annual_income,
        'Spending_Score': spending_score,
        'Monthly_Visits': monthly_visits,
        'Gender': gender,
        'Preferred_Product': preferred_product
    })
    
    # Sauvegarder les donnees generees dans le dossier resultats
    chemin_fichier = os.path.join(RESULTATS_DIR, "donnees_clients_generes.csv")
    data.to_csv(chemin_fichier, index=False)
    print(f"Donnees synthetiques generees et sauvegardees dans '{chemin_fichier}'")
    
    return data

def preprocess_data(data):
    """Pretraitement des donnees"""
    print("\n" + "="*60)
    print("PRETRAITEMENT DES DONNEES")
    print("="*60)
    
    data_clean = data.copy()
    
    # Informations initiales
    print(f"Nombre initial de clients : {len(data_clean)}")
    
    # Gestion des valeurs manquantes
    valeurs_manquantes = data_clean.isnull().sum().sum()
    if valeurs_manquantes > 0:
        print(f"Suppression de {valeurs_manquantes} valeurs manquantes")
        data_clean = data_clean.dropna()
        print(f"Nombre de clients apres nettoyage : {len(data_clean)}")
    else:
        print("Aucune valeur manquante detectee")
    
    # Encodage des variables categoriques
    le_gender = LabelEncoder()
    le_product = LabelEncoder()
    
    data_clean['Gender_encoded'] = le_gender.fit_transform(data_clean['Gender'])
    data_clean['Preferred_Product_encoded'] = le_product.fit_transform(data_clean['Preferred_Product'])
    
    print("Pretraitement termine")
    
    return data_clean, le_gender, le_product

def perform_clustering(data, n_clusters=4):
    """Executer l'algorithme K-Means"""
    print(f"\nExecution du clustering K-Means avec {n_clusters} clusters...")
    
    features = ["Age", "Annual_Income", "Spending_Score", "Monthly_Visits"]
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Scores d'evaluation
    silhouette = silhouette_score(X_scaled, clusters)
    db_index = davies_bouldin_score(X_scaled, clusters)
    
    # Ajouter les clusters aux donnees
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters
    
    return {
        'scaler': scaler,
        'kmeans': kmeans,
        'clusters': clusters,
        'features_scaled': X_scaled,
        'silhouette_score': silhouette,
        'davies_bouldin_score': db_index,
        'features': features,
        'data_with_clusters': data_with_clusters
    }

def create_cluster_profiles(data, clusters):
    """Creer des profils detailles pour chaque cluster"""
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters
    
    profiles = {}
    
    for cluster in sorted(data_with_clusters['Cluster'].unique()):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
        
        profile = {
            'Taille': len(cluster_data),
            'Pourcentage': len(cluster_data) / len(data_with_clusters) * 100,
            'Age_moyen': cluster_data['Age'].mean(),
            'Revenu_moyen': cluster_data['Annual_Income'].mean(),
            'Score_depense_moyen': cluster_data['Spending_Score'].mean(),
            'Visites_moyennes': cluster_data['Monthly_Visits'].mean(),
            'Genre_majoritaire': cluster_data['Gender'].mode()[0] if not cluster_data['Gender'].mode().empty else 'N/A',
            'Genre_Female_pourcentage': (cluster_data['Gender'] == 'Female').mean() * 100,
            'Genre_Male_pourcentage': (cluster_data['Gender'] == 'Male').mean() * 100,
            'Produit_prefere': cluster_data['Preferred_Product'].mode()[0] if not cluster_data['Preferred_Product'].mode().empty else 'N/A',
            'Description': ''
        }
        
        # Determiner la description du cluster
        income = profile['Revenu_moyen']
        spending = profile['Score_depense_moyen']
        visits = profile['Visites_moyennes']
        
        if income > data['Annual_Income'].quantile(0.75) and spending > data['Spending_Score'].quantile(0.75):
            profile['Description'] = 'Clients Premium - Haute valeur'
        elif visits > data['Monthly_Visits'].quantile(0.75):
            profile['Description'] = 'Clients Fideles - Visites frequentes'
        elif spending < data['Spending_Score'].quantile(0.25):
            profile['Description'] = 'Clients Economes - Depenses faibles'
        else:
            profile['Description'] = 'Clients Standards - Comportement moyen'
        
        profiles[cluster] = profile
    
    return profiles

def analyser_donnees(data):
    """Analyse exploratoire des donnees"""
    print("\n" + "="*60)
    print("ANALYSE EXPLORATOIRE DES DONNEES")
    print("="*60)
    
    # Informations de base
    print(f"\nINFORMATIONS GENERALES")
    print(f"Nombre de clients : {len(data)}")
    print(f"Nombre de variables : {len(data.columns)}")
    
    print(f"\nAPERÇU DES DONNEES")
    print(data.head())
    
    print(f"\nSTATISTIQUES DESCRIPTIVES")
    print(data.describe())
    
    print(f"\nTYPES DE DONNEES")
    print(data.dtypes)
    
    print(f"\nVALEURS MANQUANTES")
    missing = data.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count} valeurs ({count/len(data)*100:.1f}%)")
        else:
            print(f"  {col}: Aucune valeur manquante")
    
    return data

def visualiser_donnees(data):
    """Generer les visualisations des donnees"""
    print("\nGENERATION DES VISUALISATIONS...")
    
    # Distribution des variables numeriques
    num_cols = ["Age", "Annual_Income", "Spending_Score", "Monthly_Visits"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    for idx, col in enumerate(num_cols):
        axes[idx].hist(data[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution de {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequence')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    chemin_image = os.path.join(RESULTATS_DIR, 'distribution_variables.png')
    plt.savefig(chemin_image, dpi=300, bbox_inches='tight')
    print(f"{chemin_image} genere")
    plt.close()
    
    # Analyse des variables categoriques
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Genre
    gender_counts = data['Gender'].value_counts()
    axes[0].bar(gender_counts.index, gender_counts.values, color=['lightblue', 'lightpink'])
    axes[0].set_title('Repartition par genre')
    axes[0].set_ylabel('Nombre')
    for i, v in enumerate(gender_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center')
    
    # Produits preferes
    product_counts = data['Preferred_Product'].value_counts()
    axes[1].barh(range(len(product_counts)), product_counts.values)
    axes[1].set_yticks(range(len(product_counts)))
    axes[1].set_yticklabels(product_counts.index)
    axes[1].set_title('Produits preferes')
    axes[1].set_xlabel('Nombre')
    
    plt.tight_layout()
    chemin_image = os.path.join(RESULTATS_DIR, 'analyse_categorielle.png')
    plt.savefig(chemin_image, dpi=300, bbox_inches='tight')
    print(f"{chemin_image} genere")
    plt.close()
    
    # Scatter plot revenu vs depenses
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data['Annual_Income'], data['Spending_Score'], 
                         c=data['Age'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Age')
    plt.xlabel('Revenu Annuel ($)')
    plt.ylabel('Score de Depense')
    plt.title('Revenu vs Score de Depense (couleur = Age)')
    plt.grid(True, alpha=0.3)
    chemin_image = os.path.join(RESULTATS_DIR, 'revenu_vs_depenses.png')
    plt.savefig(chemin_image, dpi=300, bbox_inches='tight')
    print(f"{chemin_image} genere")
    plt.close()
    
    # Matrice de correlation
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        corr_matrix = numeric_data.corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Matrice de correlation')
        plt.tight_layout()
        chemin_image = os.path.join(RESULTATS_DIR, 'matrice_correlation.png')
        plt.savefig(chemin_image, dpi=300, bbox_inches='tight')
        print(f"{chemin_image} genere")
        plt.close()

def executer_clustering(data_clean, n_clusters=4):
    """Executer le clustering et generer les visualisations"""
    print("\n" + "="*60)
    print("CLUSTERING K-MEANS")
    print("="*60)
    
    # Clustering
    results = perform_clustering(data_clean, n_clusters)
    data_with_clusters = results['data_with_clusters']
    
    # Metriques
    print(f"\nMETRIQUES D'EVALUATION")
    print(f"Score Silhouette : {results['silhouette_score']:.3f} (ideal > 0.5)")
    print(f"Score Davies-Bouldin : {results['davies_bouldin_score']:.3f} (ideal proche de 0)")
    print(f"Inertie : {results['kmeans'].inertia_:.0f}")
    
    # Visualisation 2D avec PCA
    print("\nGENERATION DES VISUALISATIONS DES CLUSTERS...")
    
    # PCA pour visualisation 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(results['features_scaled'])
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=data_with_clusters['Cluster'], cmap='tab10', s=50, alpha=0.6)
    plt.xlabel('Composante Principale 1')
    plt.ylabel('Composante Principale 2')
    plt.title('Visualisation 2D des clusters (PCA)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    chemin_image = os.path.join(RESULTATS_DIR, 'clusters_pca.png')
    plt.savefig(chemin_image, dpi=300, bbox_inches='tight')
    print(f"{chemin_image} genere")
    plt.close()
    
    # Revenu vs Score de depense par cluster
    plt.figure(figsize=(10, 8))
    
    clusters_unique = sorted(data_with_clusters['Cluster'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters_unique)))
    
    for i, cluster in enumerate(clusters_unique):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
        plt.scatter(cluster_data['Annual_Income'], cluster_data['Spending_Score'],
                   label=f'Cluster {cluster}', color=colors[i], s=50, alpha=0.6)
    
    # Centroides
    centroids = results['scaler'].inverse_transform(results['kmeans'].cluster_centers_)
    plt.scatter(centroids[:, 1], centroids[:, 2],
               marker='X', s=200, c='black', label='Centroides', edgecolors='red', linewidths=2)
    
    plt.xlabel('Revenu Annuel ($)')
    plt.ylabel('Score de Depense')
    plt.title('Segmentation Clientèle : Revenu vs Score de Depense')
    plt.legend()
    plt.grid(True, alpha=0.3)
    chemin_image = os.path.join(RESULTATS_DIR, 'clusters_revenu_depenses.png')
    plt.savefig(chemin_image, dpi=300, bbox_inches='tight')
    print(f"{chemin_image} genere")
    plt.close()
    
    # Methode du coude pour determiner le nombre optimal de clusters
    print("\nDETERMINATION DU NOMBRE OPTIMAL DE CLUSTERS...")
    
    wcss = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters_temp = kmeans_temp.fit_predict(results['features_scaled'])
        wcss.append(kmeans_temp.inertia_)
        silhouette_scores.append(silhouette_score(results['features_scaled'], clusters_temp))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(k_range, wcss, 'bo-')
    ax1.set_xlabel('Nombre de clusters')
    ax1.set_ylabel('WCSS (Inertie)')
    ax1.set_title('Methode du coude')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Nombre de clusters')
    ax2.set_ylabel('Score Silhouette')
    ax2.set_title('Score Silhouette')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chemin_image = os.path.join(RESULTATS_DIR, 'methode_coude.png')
    plt.savefig(chemin_image, dpi=300, bbox_inches='tight')
    print(f"{chemin_image} genere")
    plt.close()
    
    # Trouver le meilleur k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nNombre optimal de clusters recommande : {optimal_k}")
    
    return data_with_clusters, results

def analyser_profils(data_with_clusters, results):
    """Analyser les profils des clusters"""
    print("\n" + "="*60)
    print("ANALYSE DES PROFILS DES CLUSTERS")
    print("="*60)
    
    # Creation des profils
    profiles = create_cluster_profiles(data_with_clusters, results['clusters'])
    
    # Afficher les profils
    print("\nPROFILS DETAILLES DES CLUSTERS")
    print("="*60)
    
    for cluster, profile in profiles.items():
        print(f"\n{'='*50}")
        print(f"CLUSTER {cluster} - {profile['Description']}")
        print(f"{'='*50}")
        print(f"Statistiques :")
        print(f"  * Taille : {profile['Taille']} clients ({profile['Pourcentage']:.1f}%)")
        print(f"  * Age moyen : {profile['Age_moyen']:.1f} ans")
        print(f"  * Revenu moyen : ${profile['Revenu_moyen']:,.0f}")
        print(f"  * Score depense moyen : {profile['Score_depense_moyen']:.1f}/100")
        print(f"  * Visites/mois : {profile['Visites_moyennes']:.1f}")
        
        print(f"\nDemographie :")
        print(f"  * Genre majoritaire : {profile['Genre_majoritaire']}")
        print(f"  * Femmes : {profile['Genre_Female_pourcentage']:.1f}%")
        print(f"  * Hommes : {profile['Genre_Male_pourcentage']:.1f}%")
        print(f"  * Produit prefere : {profile['Produit_prefere']}")
        
        print(f"\nRecommandations marketing :")
        if "Premium" in profile['Description']:
            print("  1. Offres VIP et produits exclusifs")
            print("  2. Service client personnalise")
            print("  3. Invitations a des evenements speciaux")
            print("  4. Programmes de fidelite premium")
        elif "Fideles" in profile['Description']:
            print("  1. Programmes de fidelite avec recompenses")
            print("  2. Offres de cross-selling")
            print("  3. Communication reguliere (newsletters)")
            print("  4. Programmes de parrainage")
        elif "Economes" in profile['Description']:
            print("  1. Promotions ciblees")
            print("  2. Offres de premiere commande")
            print("  3. Produits en soldes")
            print("  4. Echantillons gratuits")
        else:
            print("  1. Campagnes marketing generales")
            print("  2. Offres groupees")
            print("  3. Communication moderee")
            print("  4. Enquetes de satisfaction")
    
    # Vue comparative
    print(f"\n{'='*60}")
    print("TABLEAU COMPARATIF DES CLUSTERS")
    print(f"{'='*60}")
    
    comparative_df = pd.DataFrame(profiles).T
    
    # Formater l'affichage
    display_df = comparative_df[['Taille', 'Pourcentage', 'Age_moyen', 'Revenu_moyen', 
                               'Score_depense_moyen', 'Visites_moyennes', 'Description']].copy()
    
    # Renommer les colonnes pour un meilleur affichage
    display_df.columns = ['Taille', '%', 'Age moy', 'Revenu moy ($)', 'Score dep', 'Visites/mois', 'Description']
    
    # Formater les nombres
    pd.set_option('display.float_format', lambda x: f'{x:.1f}')
    
    print(display_df.to_string())
    
    # Sauvegarder les resultats dans le dossier resultats
    chemin_fichier = os.path.join(RESULTATS_DIR, 'resultats_clustering.csv')
    data_with_clusters.to_csv(chemin_fichier, index=False)
    print(f"\nResultats sauvegardes dans '{chemin_fichier}'")
    
    # Visualisation comparative
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Taille des clusters
    sizes = [profile['Taille'] for profile in profiles.values()]
    labels = [f'Cluster {i}' for i in profiles.keys()]
    axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Repartition des clusters')
    
    # Revenu moyen par cluster
    incomes = [profile['Revenu_moyen'] for profile in profiles.values()]
    axes[0, 1].bar(range(len(incomes)), incomes)
    axes[0, 1].set_xticks(range(len(incomes)))
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_title('Revenu moyen par cluster')
    axes[0, 1].set_ylabel('Revenu ($)')
    
    # Score de depense moyen par cluster
    scores = [profile['Score_depense_moyen'] for profile in profiles.values()]
    axes[1, 0].bar(range(len(scores)), scores)
    axes[1, 0].set_xticks(range(len(scores)))
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_title('Score de depense moyen par cluster')
    axes[1, 0].set_ylabel('Score')
    
    # Visites moyennes par cluster
    visits = [profile['Visites_moyennes'] for profile in profiles.values()]
    axes[1, 1].bar(range(len(visits)), visits)
    axes[1, 1].set_xticks(range(len(visits)))
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_title('Visites moyennes par cluster')
    axes[1, 1].set_ylabel('Visites/mois')
    
    plt.tight_layout()
    chemin_image = os.path.join(RESULTATS_DIR, 'comparaison_clusters.png')
    plt.savefig(chemin_image, dpi=300, bbox_inches='tight')
    print(f"{chemin_image} genere")
    plt.close()
    
    return profiles

def predire_nouveau_client(data_clean, results, profiles):
    """Predire le cluster pour un nouveau client"""
    print("\n" + "="*60)
    print("PREDICTION POUR UN NOUVEAU CLIENT")
    print("="*60)
    
    # Exemple de nouveau client
    print("\nEXEMPLE DE NOUVEAU CLIENT :")
    
    # Creer un exemple de client
    customer_features = {
        "Age": 35,
        "Annual_Income": 75000,
        "Spending_Score": 75,
        "Monthly_Visits": 8,
        "Gender": "Male",
        "Preferred_Product": "Electronics"
    }
    
    print(f"  * Age : {customer_features['Age']}")
    print(f"  * Revenu annuel : ${customer_features['Annual_Income']:,.0f}")
    print(f"  * Score de depense : {customer_features['Spending_Score']}/100")
    print(f"  * Visites mensuelles : {customer_features['Monthly_Visits']}")
    print(f"  * Genre : {customer_features['Gender']}")
    print(f"  * Produit prefere : {customer_features['Preferred_Product']}")
    
    # Preparer les donnees pour la prediction
    customer_df = pd.DataFrame([customer_features])
    
    # Selection des features
    features = ["Age", "Annual_Income", "Spending_Score", "Monthly_Visits"]
    X_customer = customer_df[features]
    
    # Normalisation
    X_scaled = results['scaler'].transform(X_customer)
    
    # Prediction
    cluster = results['kmeans'].predict(X_scaled)[0]
    
    # Calcul de la distance aux centroides
    distances = results['kmeans'].transform(X_scaled)[0]
    confidence = 1 - (distances[cluster] / distances.max())
    
    print(f"\nRESULTAT DE LA PREDICTION :")
    print(f"Ce client appartient au Cluster {cluster}")
    print(f"Confiance de prediction : {confidence*100:.1f}%")
    
    if cluster in profiles:
        profile = profiles[cluster]
        print(f"\nCARACTERISTIQUES DU CLUSTER {cluster} :")
        print(f"Description : {profile['Description']}")
        print(f"Taille : {profile['Taille']} clients ({profile['Pourcentage']:.1f}%)")
    
    return customer_features, cluster, confidence

def sauvegarder_rapport(data_clean, results, profiles, n_clusters):
    """Sauvegarder un rapport complet des resultats"""
    print("\n" + "="*60)
    print("GENERATION DU RAPPORT COMPLET")
    print("="*60)
    
    chemin_rapport = os.path.join(RESULTATS_DIR, "rapport_clustering.txt")
    
    with open(chemin_rapport, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT DE SEGMENTATION DE CLIENTELE\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. RESUME EXECUTIF\n")
        f.write("-"*40 + "\n")
        f.write(f"Date d'analyse : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Nombre total de clients : {len(data_clean)}\n")
        f.write(f"Nombre de clusters : {n_clusters}\n")
        f.write(f"Score Silhouette : {results['silhouette_score']:.3f}\n")
        f.write(f"Score Davies-Bouldin : {results['davies_bouldin_score']:.3f}\n\n")
        
        f.write("2. REPARTITION DES CLUSTERS\n")
        f.write("-"*40 + "\n")
        cluster_counts = results['data_with_clusters']['Cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            percentage = count / len(data_clean) * 100
            desc = profiles[cluster]['Description'] if cluster in profiles else "N/A"
            f.write(f"Cluster {cluster}: {count} clients ({percentage:.1f}%) - {desc}\n")
        f.write("\n")
        
        f.write("3. PROFILS DETAILS DES CLUSTERS\n")
        f.write("-"*40 + "\n")
        for cluster, profile in profiles.items():
            f.write(f"\n[CLUSTER {cluster}] - {profile['Description']}\n")
            f.write(f"  Taille : {profile['Taille']} clients ({profile['Pourcentage']:.1f}%)\n")
            f.write(f"  Age moyen : {profile['Age_moyen']:.1f} ans\n")
            f.write(f"  Revenu moyen : ${profile['Revenu_moyen']:,.0f}\n")
            f.write(f"  Score depense : {profile['Score_depense_moyen']:.1f}/100\n")
            f.write(f"  Visites/mois : {profile['Visites_moyennes']:.1f}\n")
            f.write(f"  Genre majoritaire : {profile['Genre_majoritaire']}\n")
            f.write(f"  Produit prefere : {profile['Produit_prefere']}\n")
        f.write("\n")
        
        f.write("4. RECOMMANDATIONS MARKETING\n")
        f.write("-"*40 + "\n")
        for cluster, profile in profiles.items():
            f.write(f"\nCluster {cluster} ({profile['Description']}):\n")
            if "Premium" in profile['Description']:
                f.write("  - Offres VIP et produits exclusifs\n")
                f.write("  - Service client personnalise\n")
                f.write("  - Invitations evenements speciaux\n")
            elif "Fideles" in profile['Description']:
                f.write("  - Programmes de fidelite\n")
                f.write("  - Offres de cross-selling\n")
                f.write("  - Communication reguliere\n")
            elif "Economes" in profile['Description']:
                f.write("  - Promotions ciblees\n")
                f.write("  - Offres de premiere commande\n")
                f.write("  - Produits en soldes\n")
            else:
                f.write("  - Campagnes generales\n")
                f.write("  - Offres groupees\n")
                f.write("  - Communication moderee\n")
    
    print(f"Rapport complet sauvegarde dans '{chemin_rapport}'")

# =========================
# 4. EXECUTION PRINCIPALE
# =========================
def main():
    """Fonction principale - execution complete du pipeline"""
    print("\n" + "="*70)
    print("SEGMENTATION DE CLIENTELE PAR MACHINE LEARNING (K-MEANS)")
    print("="*70)
    
    # Verifier que le dossier resultats existe
    if not os.path.exists(RESULTATS_DIR):
        print(f"\nCreation du dossier '{RESULTATS_DIR}'...")
        os.makedirs(RESULTATS_DIR, exist_ok=True)
    
    print(f"\nTous les resultats seront sauvegardes dans le dossier : {RESULTATS_DIR}/")
    
    # Parametres
    n_clusters = 4  # Nombre de clusters
    
    print(f"\nDEMARRAGE DE L'ANALYSE...")
    print(f"Nombre de clusters : {n_clusters}")
    
    # Etape 1: Charger les donnees
    print(f"\n{'='*50}")
    print("ETAPE 1 : CHARGEMENT DES DONNEES")
    print(f"{'='*50}")
    data = charger_donnees()
    
    # Etape 2: Analyse exploratoire
    print(f"\n{'='*50}")
    print("ETAPE 2 : ANALYSE EXPLORATOIRE")
    print(f"{'='*50}")
    analyser_donnees(data)
    visualiser_donnees(data)
    
    # Etape 3: Pretraitement
    print(f"\n{'='*50}")
    print("ETAPE 3 : PRETRAITEMENT")
    print(f"{'='*50}")
    data_clean, _, _ = preprocess_data(data)
    
    # Etape 4: Clustering
    print(f"\n{'='*50}")
    print("ETAPE 4 : CLUSTERING K-MEANS")
    print(f"{'='*50}")
    data_with_clusters, results = executer_clustering(data_clean, n_clusters)
    
    # Etape 5: Analyse des profils
    print(f"\n{'='*50}")
    print("ETAPE 5 : ANALYSE DES PROFILS")
    print(f"{'='*50}")
    profiles = analyser_profils(data_with_clusters, results)
    
    # Etape 6: Prediction exemple
    print(f"\n{'='*50}")
    print("ETAPE 6 : PREDICTION (EXEMPLE)")
    print(f"{'='*50}")
    predire_nouveau_client(data_clean, results, profiles)
    
    # Etape 7: Sauvegarder le rapport complet
    sauvegarder_rapport(data_clean, results, profiles, n_clusters)
    
    # Resume final
    print(f"\n{'='*70}")
    print("ANALYSE TERMINEE AVEC SUCCES !")
    print(f"{'='*70}")
    
    # Lister les fichiers generes
    fichiers_generes = os.listdir(RESULTATS_DIR)
    fichiers_images = [f for f in fichiers_generes if f.endswith('.png')]
    fichiers_csv = [f for f in fichiers_generes if f.endswith('.csv')]
    fichiers_txt = [f for f in fichiers_generes if f.endswith('.txt')]
    
    print(f"\nFICHIERS GENERES DANS LE DOSSIER '{RESULTATS_DIR}/' :")
    
    if fichiers_images:
        print(f"\n VISUALISATIONS ({len(fichiers_images)} fichiers):")
        for img in sorted(fichiers_images):
            print(f"  • {img}")
    
    if fichiers_csv:
        print(f"\n DONNEES ({len(fichiers_csv)} fichiers):")
        for csv in sorted(fichiers_csv):
            print(f"  • {csv}")
    
    if fichiers_txt:
        print(f"\nRAPPORTS ({len(fichiers_txt)} fichiers):")
        for txt in sorted(fichiers_txt):
            print(f"   {txt}")
    
    print(f"\nRESUME DES RESULTATS :")
    print(f"* Nombre de clients analyses : {len(data_clean)}")
    print(f"* Nombre de clusters crees : {n_clusters}")
    print(f"* Score Silhouette : {results['silhouette_score']:.3f}")
    
    # Afficher les clusters
    cluster_counts = data_with_clusters['Cluster'].value_counts().sort_index()
    print(f"\nREPARTITION DES CLUSTERS :")
    for cluster, count in cluster_counts.items():
        percentage = count / len(data_with_clusters) * 100
        if cluster in profiles:
            description = profiles[cluster]['Description']
            print(f"  Cluster {cluster}: {count} clients ({percentage:.1f}%) - {description}")
        else:
            print(f"  Cluster {cluster}: {count} clients ({percentage:.1f}%)")
    
    print(f"\n{'='*70}")
    print("Analyse terminee avec succes !")
    print(f"Tous les resultats sont disponibles dans le dossier '{RESULTATS_DIR}/'")
    print(f"{'='*70}")

# =========================
# 5. EXECUTION
# =========================
if __name__ == "__main__":
    main()