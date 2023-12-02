"""
Fichier pour stocker les variables contenant les données
"""

from pandas import read_csv

MOVIES = read_csv("donnees/movies.csv")
RATINGS = read_csv("donnees/ratings.csv")

LISTE_GENRES = []

# Parcours des données
for donnee in MOVIES["genres"]:
    # Liste des genres du film courant
    genres = donnee.split("|")
    if genres != ['(no genres listed)'] :
        for genre in genres :
            if genre not in LISTE_GENRES :
                # Ajout d'un nouveau genre
                LISTE_GENRES.append(genre)
