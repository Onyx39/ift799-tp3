"""
Question 3
Création d'une matrice de contenu pour les films
"""

from pandas import DataFrame
from constantes import LISTE_GENRES
from donnees_calculees import MOVIES1

def creation_matrice_films () :
    """
    Crée une matrice bianire de contenu pour les films
    Marque l'appartenance d'un film à des genres cinématographique

    Aucune entrée

    Aucune sortie
    """


    # Creation d'une matrice vide
    matrice = DataFrame(None, columns=LISTE_GENRES, index=MOVIES1["movieId"])

    # Remplissage de la matrice avec des 0
    matrice.fillna(value=0, inplace=True)

    # Parcours des données
    for movie_index in MOVIES1["movieId"] :
        # Liste des genres du film courant
        genres = MOVIES1.loc[MOVIES1[MOVIES1["movieId"] == movie_index].index, "genres"].item()
        genres = genres.split('|')

        # Pour chaque genre, modification de la matrice
        for genre in genres :
            matrice.loc[movie_index, genre] = 1

    matrice.to_csv("donnees/movies_matrix.csv")

    return True
