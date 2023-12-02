
from pandas import DataFrame, read_csv


MOVIES_MATRIX = DataFrame(read_csv("donnees/movies_matrix.csv"))

truc = film = MOVIES_MATRIX.loc[57357]
print(truc)