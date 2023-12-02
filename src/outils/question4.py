"""
Question 4
Création de la matrice de profil des utilisateurs
"""

from pandas import DataFrame

from constantes import LISTE_GENRES
from donnees_calculees import RATINGS1, MOVIES_MATRIX



def creation_matrice_utilisateurs () :
    """
    Crée la matrice de stockage des profils utilisateurs

    Aucune entrée

    Auncune sortie
    """

    # Définition de l'intervalle des ID utilisateurs
    min_id = min(RATINGS1["userId"])
    max_id = max(RATINGS1["userId"])
    print(min_id, max_id)

    user_id_range = list(range (min_id, max_id + 1))
    matrice_utilisateurs = DataFrame(None, columns=LISTE_GENRES, index=user_id_range)


    for user_id in range (min_id, max_id + 1) :
        print('user_id : ', user_id)
        donnees = RATINGS1[RATINGS1["userId"] == user_id]
        profil = [0]*19
        # print(profil)
        # print(donnees.info)
        # print(MOVIES_MATRIX.info)
        for index in donnees.T :
            id_film = RATINGS1.loc[index, "movieId"]
            notation = RATINGS1.loc[index, "rating"]
            print(id_film, notation)
            break
            film = MOVIES_MATRIX.loc[id_film]
            # print(film)
            # print(len(film), len(profil))
            info_utilisateur = film[1:] * notation
            # print(info_utilisateur)
            profil += info_utilisateur
        break
        # print(profil)
        matrice_utilisateurs.iloc[matrice_utilisateurs.index == user_id] = profil
        # print(matrice_utilisateurs)
        print(matrice_utilisateurs)

    matrice_utilisateurs.to_csv('donnees/user_matrix.csv')

    return True
