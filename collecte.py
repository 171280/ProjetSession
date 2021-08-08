# Importation des differentes librairies
import numpy as np
import pandas as pd
import seaborn as sns

# Nous allons lire les données du fichier csv et les importer dans notre dataframe qui s'appelle dfClientsPerdus.
dfClientsPerdus = pd.read_csv("BankChurners.csv")

# Nous allons chercher le nombre de descripteurs et d'individus de notre dataframe
print(f"Le nombre d'individus est de: {dfClientsPerdus.shape[0]}")
print(f"Le nombre de prédicteurs est de: {dfClientsPerdus.shape[1]}")
print("Notre dataframe contient 10127 individus et 23 prédicteurs")

# Nous allons faire une description de nos descripteurs
dfClientsPerdus.info()
dfClientsPerdus.describe()
dfClientsPerdus.describe(include='O')

print("Nous avons 23 prédicteurs")
print("Nous avons 17 prédicteurs qui sont des variables quantitatives")
print("Nous avons 06 prédicteurs qui sont des variables qualitatives")

#sauvegarder le data
dfClientsPerdus.to_csv("BankChurners_collecte.csv",index=False)

print("La collecte de données est terminée.")
