"""
Question 6
Validation du système de recommandations
"""

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame, read_csv
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, train_test_split

def separer_donnees(ratings, seed):

    # On fait une stratification des ensembles de données pour garder les même proportions
    ## Division en ensembles d'entraînement (60%) et de temp = test/validation (40%)
    ratings_train, temp = train_test_split(ratings, test_size=0.4, stratify=ratings['rating'], random_state=seed)

    ## Division de l'ensemble temp en ensembles de validation et de test (50% chacun, donc 20% de l'ensemble original chacun)
    ratings_test, ratings_evaluation = train_test_split(temp, test_size=0.5, stratify=temp['rating'], random_state=seed)

    # Sauvegarde des ensembles dans des fichiers CSV
    ratings_train.to_csv('donnees/question6/ratings_train.csv', index=False)
    ratings_evaluation.to_csv('donnees/question6/ratings_evaluation.csv', index=False)
    ratings_test.to_csv('donnees/question6/ratings_test.csv', index=False)

def creation_entrees_classificateur(result_path, dataset_path, clustering_path, profils):
    
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
        current_cluster = set_clusters.loc[i, 'cluster']
        
        # On sélectionne les voisins ayant le même cluster
        neighbors_cluster_df = set_clusters[set_clusters['cluster'] == current_cluster]
        
        # Parmi ces voisins de cluster, on sélectionne ceux qui ont noté le film voulu
        neighbors_cluster_movie_df = neighbors_cluster_df[neighbors_cluster_df['movieId'] == current_movie_id]
        
        neighbors_ratings = chercher_5_plus_proches_voisins(profils, neighbors_cluster_movie_df, current_user_id)
        
        data['userId'].append(current_user_id)
        data['movieId'].append(current_movie_id)
        data['userRating'].append(current_rating)
        
        for index in range(len(neighbors_ratings)):
            data[f'neighborRating{index + 1}'].append(neighbors_ratings[index])
        for index in range(len(neighbors_ratings), 5):
            data[f'neighborRating{index + 1}'].append(sum(neighbors_ratings) / len(neighbors_ratings))

    df = DataFrame(data)
    df.to_csv(result_path)   
        
def chercher_5_plus_proches_voisins(profils, cluster_neighbors, user_id):
    closest_neighbors_ratings = [] # contiendra l'id de chaque voisin sélectionné
    distances = [] # distances[i] contiendra les distances entre neighbors[i] et user_id
    
    for j in cluster_neighbors.index:
        current_neighbor_id = cluster_neighbors.loc[j, 'userId'] + 1

        if (current_neighbor_id != user_id):
            dist_user_i = np.linalg.norm((profils.loc[user_id-1]).to_numpy() - (profils.loc[current_neighbor_id-1]).to_numpy())
                
            # S'il y a moins de 5 voisins enregistrés, on ajoute le voisin courant
            if len(closest_neighbors_ratings) < 5:
                closest_neighbors_ratings.append(cluster_neighbors.loc[j, 'rating'])
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
                        
                    closest_neighbors_ratings.append(cluster_neighbors.loc[j, 'rating'])
                    distances.append(dist_user_i)
                
    return closest_neighbors_ratings


class classificateur :
    def __init__(self, model):
        self.model = model
        self.parameters = None
    
    def separer_donnees_etiquettes(self, set):
        X = set[['userId', 'movieId', 'neighborRating1', 'neighborRating2', 'neighborRating3', 'neighborRating4', 'neighborRating5']]
        Y = set[['userRating']]
        
        return X, Y
    
    def entrainer(self, X, Y):
        self.model.fit(X, Y)

    def chercher_hyperparametres(self, X, Y, seed):
        param = {
            'criterion': ['log_loss'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        grid_search = RandomizedSearchCV(estimator=self.model, param_distributions=param, cv=cv, scoring='f1', random_state=seed)

        grid_search.fit(X, Y)
        best_model = grid_search.best_estimator_
        print(f'Les meilleurs paramètres trouvés sont : {grid_search.best_params_}')
        
        return best_model

    def predire(self, X):
        Y = self.model.predict(X)
        return Y

    
    def evaluer(self, X, Y):
        Y_pred = self.predire(X)
        
        # Calcul du score F1
        f1_res = f1_score(y_true=Y, y_pred=Y_pred, average='weighted', labels=Y_pred)
        print(f'Score F1 du modèle : {f1_res}')
        
        # Tracé de la matrice de confusion
        confusion_mtrx = confusion_matrix(Y, Y_pred)
        print(confusion_mtrx)