"""
Importation des donnees calculées
"""

from pandas import DataFrame, read_csv

MOVIES_MATRIX = DataFrame(read_csv("donnees/movies_matrix.csv"))
MOVIES1 = DataFrame(read_csv("donnees/movies1.csv"))
RATINGS1 = DataFrame(read_csv("donnees/ratings1.csv"))
