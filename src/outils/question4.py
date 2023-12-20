"""
Question 4
Création de la matrice de profil des utilisateurs
"""

from pandas import DataFrame

from constantes import LISTE_GENRES


def creation_matrice_utilisateurs (movies_matrix : DataFrame,
                                   ratings1 : DataFrame,
                                   movies1 : DataFrame) :
    """
    Crée la matrice de stockage des profils utilisateurs

    Paramètres :
        movies_matrix (pd.DataFrame) : la matrice de contenu des films
        ratings1 (pd.DataFrame) : la matrice modifiée des évaluations de films
        movies1 (ps.DataFrame) : la matrice des films

    Auncune sortie
    """

    # Définition de l'intervalle des ID utilisateurs
    min_id = min(ratings1["userId"])
    max_id = max(ratings1["userId"])

    # Initialisation de la matrice de utilisateurs
    matrice_utilisateurs = DataFrame(data=None,
                                     columns=LISTE_GENRES,
                                     index=list(range (min_id, max_id + 1)))

    # Parcours des ID utilisateurs
    for user_id in list(range (min_id, max_id + 1)) :
        print(str(round(user_id/max_id*100, 2)) + "%")

        # Extraction des votes de l'utilisateur courant
        donnees = ratings1[ratings1["userId"] == user_id]

        # Initialiation du profil de l'utiliateur courant
        profil = [0]*len(LISTE_GENRES)

        # Parcours des votes de l'utilisateur courant
        for index in donnees.T :

            # Rechehe de l'ID film correspondant à celui dans 'movies_matrix.npz'
            id_film = ratings1.loc[index, "movieId"]
            id_film = str(movies1.index[movies1["movieId"] == id_film])
            id_film = id_film.split('[')[1].split(']')[0]

            # Recherche de la note
            notation = ratings1.loc[index, "rating"]

            # Extraction des genres du film noté
            ligne_film = movies_matrix.getrow(id_film)

            # Convertion des genres du film noté en DataFrame
            film = DataFrame(ligne_film.toarray())

            # Création de l'ajout à effctuer pour l'utilisateur courant
            info_utilisateur = film * notation

            # Transformation des float en int
            info_utilisateur = [int(i) for i in info_utilisateur.loc[0]]

            # Ajout des nouvelles informations au profil de l'utilisateur
            profil = [x + y for x, y in zip(profil, info_utilisateur)]

        # Mise à jour de la matrice des utilisateurs
        matrice_utilisateurs.iloc[matrice_utilisateurs.index == user_id] = profil

    # Sauvegarde de la matrice des utilisateurs
    matrice_utilisateurs.to_csv('donnees/user_matrix.csv', index=False)

    return True
