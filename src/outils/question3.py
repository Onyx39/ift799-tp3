"""
Question 3
Création d'une matrice de contenu pour les films
"""

from pandas import DataFrame, read_csv

# Importation des nouvelles données
MOVIES1 = DataFrame(read_csv("donnees/movies1.csv"))

def creation_matrice_films () :
    """
    Crée une matrice bianire de contenu pour les films
    Marque l'appartenance d'un film à des genres cinématographique

    Aucune entrée

    Aucune sortie
    """

    ###########################
    # Récupération des genres #
    ###########################

    liste_genres = []

    # Parcours des données
    for donnee in MOVIES1["genres"]:
        # Liste des genres du film courant
        genres = donnee.split("|")
        if genres != ['(no genres listed)'] :
            for genre in genres :
                if genre not in liste_genres :
                    # Ajout d'un nouveau genre
                    liste_genres.append(genre)


    ##########################
    # Création de la matrice #
    ##########################

    # Creation d'une matrice vide
    matrice = DataFrame(None, columns=liste_genres, index=MOVIES1["movieId"])

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
