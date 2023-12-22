"""
Question 6
Validation du système de recommandations
"""

from heapq import merge
import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame, read_csv
from sklearn import metrics
from sklearn.metrics import confusion_matrix, euclidean_distances, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def separer_donnees(ratings):
    seed = 42
    np.random.seed(seed)

    # On fait une stratification des ensembles de données pour garder les même proportions
    ## Division en ensembles d'entraînement (60%) et de temp = test/validation (40%)
    ratings_train, temp = train_test_split(ratings, test_size=0.4, stratify=ratings['rating'], random_state=seed)

    ## Division de l'ensemble temp en ensembles de validation et de test (50% chacun, donc 20% de l'ensemble original chacun)
    ratings_test, ratings_evaluation = train_test_split(temp, test_size=0.5, stratify=temp['rating'], random_state=seed)

    # Sauvegarde des ensembles dans des fichiers CSV
    ratings_train.to_csv('donnees/question6/ratings_train.csv', index=False)
    ratings_evaluation.to_csv('donnees/question6/ratings_evaluation.csv', index=False)
    ratings_test.to_csv('donnees/question6/ratings_test.csv', index=False)

def creation_entrees_classificateur(dataset_path, clustering_path, profils, set):
    
    set = read_csv(dataset_path)
    clusters_raw = read_csv(clustering_path)
    
    # On ajoute une colonne 'cluster' dans le set
    clusters_series = clusters_raw['cluster']
    clusters = clusters_series.to_frame()
    clusters.insert(loc=0, column='userId', value=clusters.index + 1)
    set_clusters = set.merge(clusters, on='userId', how='left')
    
    data = {'userId' : [],
            'movieId': [],
            'userRating': [],
            'neighborRating1': [],
            'neighborRating2': [],
            'neighborRating3': [],
            'neighborRating4': [],
            'neighborRating5': []}
    
    # Pour chaque utilisateur de l'ensemble set (train, validation ou test), on établit les 5 voisins
    n = len(set_clusters)
    for i in range(n):
        # On récupère l'id utilisateur et film de la ligne courante
        current_user_id = set_clusters.loc[i, 'userId']
        current_movie_id = set_clusters.loc[i, 'movieId']
        current_rating = set_clusters.loc[i, 'rating']
        
        # On sélectionne les voisins ayant le même cluster
        neighbors_cluster_df = set_clusters[set_clusters['userId'] == current_user_id]
        
        # Parmi ces voisins de cluster, on sélectionne ceux qui ont noté le film voulu
        neighbors_cluster_movie_df = neighbors_cluster_df[neighbors_cluster_df['movieId'] == current_movie_id]
        
        neighbors_ratings = chercher_5_plus_proches_voisins(profils, neighbors_cluster_movie_df, current_user_id)
        
        data['userId'].append(current_user_id)
        data['movieId'].append(current_movie_id)
        data['userRating'].append(current_rating)
        
        for index in range(5):
            data[f'neighborRating{index}'].append(neighbors_ratings[index])
    df = DataFrame(data)
    df.to_csv('donnees/question6/entrees_classificateur.csv', index=False)   
        
def chercher_5_plus_proches_voisins(profils, cluster_neighbors, user_id):
    closest_neighbors_ratings = [] # contiendra l'id de chaque voisin sélectionné
    distances = [] # distances[i] contiendra les distances entre neighbors[i] et user_id
        
    nb_neighbors = len(cluster_neighbors)
    for j in range(nb_neighbors):
        current_neighbor_id = cluster_neighbors[j, 'userId']
            
        dist_user_i = euclidean_distances(profils[user_id-1], profils[current_neighbor_id-1])
            
        # S'il y a moins de 5 voisins enregistrés, on ajoute le voisin courant
        if len(closest_neighbors_ratings) < 5:
            closest_neighbors_ratings.append(cluster_neighbors[j, 'rating'])
            distances.append(dist_user_i)
        else:
            # Sinon, on cherche le voisin enregistré le plus distant de user_id
            dist_max = max(distances)
            index_max = distances.index(dist_max)
                
            # Si le voisin courant est plus proche de user_id,
            # on supprime le voisin enregistré le plus distant
            # et on ajoute le voisin courant
            if dist_user_i < dist_max:
                    
                _ = distances.pop(index_max)
                _ = closest_neighbors_ratings.pop(index_max)
                    
                closest_neighbors_ratings.append(cluster_neighbors[j, 'rating'])
                distances.append(dist_user_i)
                
    return closest_neighbors_ratings

def entrainer():
    return

def chercher_hyperparametres():
    return

def predire():
    return

def evaluer():
    return

def classificateur(films, votes):
    graine = 42
    np.random.seed(graine)
    
    modele = DecisionTreeClassifier(criterion='log_loss', random_state=graine)
    modele.fit(films, votes)
    predictions = modele.predict(films)


    # Affichage d'une matrice de confusion
    
    plt.figure()
    confusion_mtrx = metrics.confusion_matrix(votes, predictions)
    group_names = ['TrueNeg', 'FalsePos', 'FalseNeg','TruePos']
    group_counts = ["{0: 0.0f}".format(value)
    for value in confusion_mtrx.flatten()]
    group_percentages =['{0:.2%}'.format(value) for value in confusion_mtrx.flatten() / np.sum(confusion_mtrx)]
    labels =[f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    ax = plt.axes()
    sns.heatmap(confusion_mtrx, annot=labels, fmt='', cmap='Blues')
    ax.set_title("Matrice de confusion du modèle")


    # Affichage du score F1
    
    f1_res = f1_score(votes, predictions, average='weighted', labels=predictions)
    print(f'Score F1 du modèle : {f1_res}')

    