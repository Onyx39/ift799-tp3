"""
Fichier pour stocker les variables contenant les données
"""

from pandas import read_csv

MOVIES = read_csv("donnees/movies.csv")
RATINGS = read_csv("donnees/ratings.csv")
