"""
Question 6
Validation du système de recommandations
"""

import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from pandas import read_csv
from sklearn import metrics
from sklearn.metrics import confusion_matrix, euclidean_distances, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def split():
    seed = 42
    np.random.seed(seed)
    
    # Charger les données
    ratings = read_csv('donnees/ratings1.csv')

    # On fait une stratification des ensembles de données pour garder les même proportions
    ## Division en ensembles d'entraînement (60%) et de temp = test/validation (40%)
    ratings_train, temp = train_test_split(ratings, test_size=0.4, stratify=ratings['rating'], random_state=seed)

    ## Division de l'ensemble temp en ensembles de validation et de test (50% chacun, donc 20% de l'ensemble original chacun)
    ratings_test, ratings_evaluation = train_test_split(temp, test_size=0.5, stratify=temp['rating'], random_state=seed)

    # Sauvegarde des ensembles dans des fichiers CSV
    ratings_train.to_csv('donnees/question6/ratings_train.csv', index=False)
    ratings_evaluation.to_csv('donnees/question6/ratings_evaluation.csv', index=False)
    ratings_test.to_csv('donnees/question6/ratings_test.csv', index=False)

split()



    