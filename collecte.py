# Importation des differentes librairies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

dfClientsPerdus = pd.read_csv("BankChurners_collecte.csv")

# Pour commencer, nous allons modiifer le nom des deux dernièrs prédicteurs car ils sont trop longs.
dfClientsPerdus.rename(columns={'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1' :\
                                'NB_mon_1'}, inplace=True)
dfClientsPerdus.rename(columns={'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2' :\
                                'NB_mon_2'}, inplace=True)

dfClientsPerdus.info()

print("Avant de commencer l'analyse exploratoire, nous allons supprimer certains prédicteurs.")
print("1- Le prédicteur CLIENTNUM sera supprimé car il est le numéro unique de chaque client")
print("2- Les prédicteurs NB_mon_1 et NB_mon_2 seront supprimés")
