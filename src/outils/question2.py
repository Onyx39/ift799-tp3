"""
Question 2
Création de nouveaux fichiers contenant les films possédant un genre ou plus
"""

import warnings

from pandas import DataFrame

from constantes import MOVIES, RATINGS

# Ignorer les warnings dans la console
warnings.filterwarnings("ignore")

def creer_nouveaux_fichiers () :
    """
    Créé de nouveaux fichiers sans les films sans genres
    Enregistre les deux fichiers dans "donnees" (movies1.csv et ratings1.csv)

    Aucune entrée

    Aucun sortie
    """

    movies1 = MOVIES[MOVIES['genres'] != '(no genres listed)']
    DataFrame.to_csv(movies1, "donnees/movies1.csv")

    ratings1 = RATINGS[RATINGS['movieId'].isin(movies1['movieId'])]

    # Traitement des valeurs non entières
    ratings1['rating'].replace(0.5, 1, inplace=True)
    ratings1['rating'].replace(1.5, 1, inplace=True)
    ratings1['rating'].replace(2.5, 2, inplace=True)
    ratings1['rating'].replace(3.5, 3, inplace=True)
    ratings1['rating'].replace(4.5, 4, inplace=True)

    DataFrame.to_csv(ratings1, "donnees/ratings1.csv")

    return True
