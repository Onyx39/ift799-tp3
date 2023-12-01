"""
Question 1
Représentation du nombre de films dans chaque genre
"""

import matplotlib.pyplot as plt
from constantes import MOVIES

def histogramme_occurrence_genres () :
    """
    Créé un histogramme représentant les occurrences des genres dans le fichier movies.csv
    Sauvegarde la figure dans le dossier résultats

    Aucune entrée

    Aucune sortie
    """

    ###########################
    # COMPTAGE DES OCCURRENCES #
    ###########################

    # Initialisation d'un dictionnaire pour compter l'occurrence des genres
    dictionnaire_genres = {}

    # Parcours des données
    for donnee in MOVIES["genres"]:
        # Liste des genres du film courant
        genres = donnee.split("|")
        if genres != ['(no genres listed)'] :
            for genre in genres :
                if genre not in dictionnaire_genres :
                    # Création d'un nouveau genre
                    dictionnaire_genres[genre] = 1
                else :
                    # Incrémentation de l'occurrence d'un genre
                    dictionnaire_genres[genre] = dictionnaire_genres[genre] + 1

    #########################
    # Création de la figure #
    #########################

    # Triage du dictionnaire en fonction du nombre d'occurrence
    dictionnaire_genres = dict(sorted(dictionnaire_genres.items(),
                                      key=lambda item: item[1],
                                      reverse=True))

    plt.bar(dictionnaire_genres.keys(), dictionnaire_genres.values())
    plt.title('Répartition des genres')
    plt.xlabel('Genres')
    # Afficher les genres en diagonale pour la lisibilité
    plt.xticks(rotation=45, ha="right")
    # Augmenter la marge ingérieure pour affichage des labels correctement
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel("Nombre d'occurrence")
    plt.savefig("resultats/question1.png")

    return True
