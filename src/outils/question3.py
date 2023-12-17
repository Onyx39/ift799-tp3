"""
Question 3
Création d'une matrice de contenu pour les films
"""

from pandas import DataFrame
from scipy.sparse import csr_matrix
from numpy import savez, array
from constantes import LISTE_GENRES

def creation_matrice_films (movies1 : DataFrame) :
    """
    Crée une matrice bianire de contenu pour les films
    Marque l'appartenance d'un film à des genres cinématographique

    Parmètre :
        movies1 (pd.DataFrame) : la matrice de films modifiée

    Aucune sortie
    """


    # Creation d'une matrice vide
    matrice = DataFrame(None, columns=LISTE_GENRES, index=movies1["movieId"])

    # Remplissage de la matrice avec des 0
    matrice.fillna(value=0, inplace=True)

    # Parcours des données
    for movie_index in movies1["movieId"] :
        # Liste des genres du film courant
        genres = movies1.loc[movies1[movies1["movieId"] == movie_index].index, "genres"].item()
        genres = genres.split('|')

        # Pour chaque genre, modification de la matrice
        for genre in genres :
            matrice.loc[movie_index, genre] = 1

    # Converti le dataframe en matrice binaire légère
    matrice_legere = csr_matrix(array(matrice))

    # Sauvegarde de la matrice binaire legère
    savez('donnees/movies_matrix.npz', data=matrice_legere.data, indices=matrice_legere.indices,
            indptr=matrice_legere.indptr, shape=matrice_legere.shape)

    return True
