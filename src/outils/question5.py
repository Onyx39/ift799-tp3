"""
Question 5
Création de clusters avec l'algorithme de clustering spectral
"""

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Ignorer les warnings dans la console
warnings.filterwarnings("ignore")


def clustering_spectral (donnees : DataFrame) :
    """
    Fonction qui réalise 4 clustering spectraux
    Les clustering sont réalisés sur une portion seulement des données
    Les représentations des données sont sauvegardées dans 'resultats'

    Paramètre :
        donnees (pd.DataFrame) : les données sur lesquelles réaliser le clustering

    Auncune sortie
    """


    for nb_clusters in [2, 3, 4, 5] :
        print(nb_clusters)
        print(donnees.shape[0])
        spectral = SpectralClustering(n_clusters=nb_clusters,
                                      affinity='nearest_neighbors',
                                      random_state=0,
                                      n_jobs=-1)
        donnees['cluster'] = spectral.fit_predict(donnees)
        donnees.to_csv(f"resultats/df_spactral_{nb_clusters}.csv", index=False)

        silhouette_avg = round(silhouette_score(donnees, donnees['cluster']), 3)
        print(f"Silhouette Score (n = {nb_clusters}): {silhouette_avg}")

        visualisation_clusters(donnees, nb_clusters, donnees.shape[0], silhouette_avg)


def visualisation_clusters (dataframe : DataFrame, nb_clusters : int, sample : int, sil : float) :
    """
    Fonction pour enregistrer un graphe représentant un clustering
    Cette méthode utilise une ACP pour représenter les données en 2 dimensions
    Le graphe est sauvegardé dans le dossier 'resultats'

    Paramètres :
        dataframe (pd.DataFrame) : les données à représenter
                                   une colonne 'cluster' doit représenter les clusters
        n (int) : le nombre de clusters
        sample (int) : le nombre de données utilisées pour le clustering
        sil (int) : la silhouette du clustering

    Aucune sortie
    """

    donnees_acp = dataframe.loc[:, dataframe.columns != 'cluster']

    pca = PCA(n_components=2)
    composantes = pca.fit_transform(donnees_acp)


    composantes_df = DataFrame(data=composantes, columns=['Composante 1', 'Composante 2'])
    composantes_df.insert(0, 'cluster', dataframe['cluster'].values)

    sns.jointplot(data=composantes_df, x='Composante 1', y='Composante 2', hue='cluster')
    plt.suptitle(f"Représentation des données transformées par la méthode ACP\n\
                 (k = {nb_clusters}, sample = {sample}, silouhette = {sil})")
    plt.tight_layout()
    plt.savefig(f"resultats/spectral_{nb_clusters}")
