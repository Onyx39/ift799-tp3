"""
Question 6
Validation du système de recommandations
"""

import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from pandas import read_csv
from sklearn import metrics
from sklearn.metrics import confusion_matrix, euclidean_distances, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def split():
    seed = 42
    np.random.seed(seed)
    
    # Charger les données
    ratings = read_csv('donnees/ratings1.csv')

    # On fait une stratification des ensembles de données pour garder les même proportions
    ## Division en ensembles d'entraînement (60%) et de temp = test/validation (40%)
    ratings_train, temp = train_test_split(ratings, test_size=0.4, stratify=ratings['rating'], random_state=seed)

    ## Division de l'ensemble temp en ensembles de validation et de test (50% chacun, donc 20% de l'ensemble original chacun)
    ratings_test, ratings_evaluation = train_test_split(temp, test_size=0.5, stratify=temp['rating'], random_state=seed)

    # Sauvegarde des ensembles dans des fichiers CSV
    ratings_train.to_csv('donnees/question6/ratings_train.csv', index=False)
    ratings_evaluation.to_csv('donnees/question6/ratings_evaluation.csv', index=False)
    ratings_test.to_csv('donnees/question6/ratings_test.csv', index=False)

split()

def sql_get_cluster_neighbors(clusters_db_path, user_id, movie_id, set):
    
    conn = sqlite3.connect(clusters_db_path)
    curs = conn.cursor()

    # Disons que clusters détient les mêmes informations que ratings
    command = "SELECT * FROM clusters WHERE user_id = ? AND movie_id = ? AND set LIKE;"
    curs.execute(command, (user_id, movie_id, set))
    
    result = curs.fetchall() # la liste des lignes (tester si elle est vide)

    conn.close()
    
    return result

def sql_get_rating(ratings_db_path, user_id, movie_id, set):
    
    conn = sqlite3.connect(ratings_db_path)
    curs = conn.cursor()

    command = "SELECT rating FROM ratings WHERE user_id = ? AND movie_id = ? AND set LIKE ?;"
    curs.execute(command, (user_id, movie_id, set))
    
    result = curs.fetchone() # la liste des lignes (tester si elle est vide)

    conn.close()
    
    return result

def find_5_closest_neighbors(profils, user_id, movie_id, set):
    neighbors = [] # contiendra l'id de chaque voisin sélectionné
    distances = [] # distances[i] contiendra les distances entre neighbors[i] et user_id
    # Sélection des utilisateurs du même cluster que user et qui ont noté movie
    restreined_cluster = sql_get_cluster_neighbors('donnees/question6/db/clusters.db', user_id, movie_id, set)
    
    # Calcul de la distance entre user et chaque utilisateur du cluster restreint au film
    n = len(restreined_cluster)
    for index in range(n):
        neighbor_id = restreined_cluster[index] + 1
        dist_user_i = euclidean_distances(profils[user_id-1], profils[neighbor_id-1])
        
        # S'il y a moins de 5 voisins enregistrés, on ajoute le voisin courant
        if len(neighbor_id) < 5:
            neighbors.append(neighbor_id)
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
                _ = neighbor_id.pop(index_max)
                
                neighbors.append(neighbor_id)
                distances.append(dist_user_i)
                
    return neighbors

def prediction(profils, user_id, movie_id, set):
    neighbors = find_5_closest_neighbors(profils, user_id, movie_id, set)
    
    neighbor_ratings = []
    for neighbor in neighbors:
        neighbor_ratings.append(sql_get_rating('donnees/question6/db/rating1.db', neighbor, movie_id, set))
    
    
    

def classifier(films, votes):
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

    