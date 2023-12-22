""""
Fichier principal, exécutable
"""

from pandas import DataFrame, read_csv
from numpy import load
from scipy.sparse import csr_matrix

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
    q4.creation_matrice_utilisateurs(MOVIES_MATRIX, RATINGS1, MOVIES1)

    ##############
    # QUESTION 5 #
    ##############

    PROFILS = DataFrame(read_csv("donnees/user_matrix.csv"))
    q5.clustering_spectral(PROFILS)
    
    ##############
    # QUESTION 6 #
    ##############

    ratings = read_csv('donnees/ratings1.csv')
    q6.separer_donnees(ratings)