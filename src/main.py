""""
Fichier principal, exécutable
"""

from pandas import DataFrame, read_csv
from numpy import load
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from outils import question1 as q1
from outils import question2 as q2
from outils import question3 as q3
from outils import question4 as q4
from outils import question5 as q5
from outils import question6 as q6

if __name__ == "__main__" :

    ##############
    # QUESTION 1 #
    ##############

    q1.histogramme_occurrence_genres()

    ##############
    # QUESTION 2 #
    ##############

    q2.creer_nouveaux_fichiers()

    ##############
    # QUESTION 3 #
    ##############

    MOVIES1 = DataFrame(read_csv("donnees/movies1.csv"))
    q3.creation_matrice_films(MOVIES1)

    ##############
    # QUESTION 4 #
    ##############

    # Charger le fichier npz
    donnees_chargees = load('donnees/movies_matrix.npz')
    MOVIES_MATRIX = csr_matrix((donnees_chargees['data'], donnees_chargees['indices'],
                                    donnees_chargees['indptr']), shape=donnees_chargees['shape'])

    RATINGS1 = DataFrame(read_csv("donnees/ratings1.csv"))

    # La méthode étant longue à s'exécuter, la matrice des utiliateurs est disponibles dans ce repo
    #q4.creation_matrice_utilisateurs(MOVIES_MATRIX, RATINGS1, MOVIES1)

    ##############
    # QUESTION 5 #
    ##############

    PROFILS = DataFrame(read_csv("donnees/user_matrix.csv"))
    q5.clustering_spectral(PROFILS)
    
    ##############
    # QUESTION 6 #
    ##############

    seed = 42
    np.random.seed(seed)
    
    ratings = read_csv('donnees/ratings1.csv')
    
    # Création des trois fichiers de données
    q6.separer_donnees(ratings.iloc[:10000], seed)


    # Création des entrées du classificateur pour l'ensemble d'entrainement
    training_set_path = 'donnees/question6/ratings_train.csv'
    evaluation_set_path = 'donnees/question6/ratings_evaluation.csv'
    testing_set_path = 'donnees/question6/ratings_test.csv'
    
    clustering_path = 'resultats/df_spactral_2.csv'
    
    q6.creation_entrees_classificateur(training_set_path, clustering_path, PROFILS.loc[:, 'Adventure':'Film-Noir'])


    # Modèle de base
    
    ## Instanciation
    training_set = read_csv('donnees/question6/entrees_classificateur_train.csv')
    evaluation_set = read_csv('donnees/question6/entrees_classificateur_evaluation.csv')
    testing_set = read_csv('donnees/question6/entrees_classificateur_test.csv')
    
    model = DecisionTreeClassifier(criterion='log_loss', random_state=seed)
    classifier = q6.classificateur(model)
    
    X_train, Y_train = classifier.separer_donnees_etiquettes(training_set)
    X_eval, Y_eval = classifier.separer_donnees_etiquettes(evaluation_set)
    X_test, Y_test = classifier.separer_donnees_etiquettes(testing_set)
    
    ## Entrainement et évaluation
    classifier.entrainer(X_train, Y_train)
    classifier.evaluer(X_eval, Y_eval)


    # Recherche d'hyperparamètres
    best_classifier_estimator = classifier.chercher_hyperparametres(X_train, Y_train, seed)
    best_classifier = q6.classificateur(best_classifier_estimator)
    best_classifier.evaluer(X_eval, X_eval)

    # Test
    best_classifier.evaluer(X_test, Y_test)