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

dfClientsPerdusNet = dfClientsPerdus.drop(['NB_mon_1', 'NB_mon_2', 'CLIENTNUM'], axis=1)

dfClientsPerdusNet.info()

# Nous avons 6 descripteurs qui sont des variables qualitatives. Nous allons chercher les modalités de ces variables.

print(f"Les modalités du descripteur Attrition_Flag {dfClientsPerdusNet['Attrition_Flag'].unique()}")
print(f"Les modalités du descripteur Gender {dfClientsPerdusNet['Gender'].unique()}")
print(f"Les modalités du descripteur Education_Level {dfClientsPerdusNet['Education_Level'].unique()}")
print(f"Les modalités du descripteur Marital_Status {dfClientsPerdusNet['Marital_Status'].unique()}")
print(f"Les modalités du descripteur Income_Category {dfClientsPerdusNet['Income_Category'].unique()}")
print(f"Les modalités du descripteur Card_Category {dfClientsPerdusNet['Card_Category'].unique()}")

dfClientsPerdusNet.groupby(['Attrition_Flag'], as_index=False).agg(total=('Attrition_Flag', 'count'))

dfClientsPerdusNet.groupby(['Attrition_Flag'], as_index=False).agg(total=('Attrition_Flag', 'count')).groupby(['Attrition_Flag']).sum().plot(kind='pie', subplots=True, shadow = True,startangle=90,figsize=(5,5), autopct='%1.1f%%')

print("On constate que 16.1% des clients se désabonnent de leur carte de credit")

dfClientsPerdusNet.groupby(['Gender'], as_index=False).agg(total=('Gender', 'count')).groupby(['Gender']).sum().plot(kind='pie', subplots=True, shadow = True,startangle=90,figsize=(5,5), autopct='%1.1f%%')

print("On constate qu'il existe 52.9% des clients qui sont de sexe féminin contre 47.1% qui sont de sexe masculin")

dfClientsPerdusNet.groupby(['Gender', 'Attrition_Flag'], as_index=False).agg(total=('Gender', 'count')).groupby(['Gender', 'Attrition_Flag']).sum().plot(kind='pie', subplots=True, shadow = True,startangle=90,\
figsize=(10,10), autopct='%1.1f%%')

print("On constate que parmi les 52.9% des clients qui sont de sexe féminin, 9.2% se désabonnent contre 43.7% qui continuent d'être clients. Aussi, sur les 47.1% de clients qui sont de sexe masculin, 6.9% de clients se désabonnent contre 40.2% qui continuent d'être des clients.")

dfClientsPerdusNet.groupby(['Education_Level', 'Attrition_Flag'], as_index=False).agg(total=('Education_Level', 'count')).groupby(['Education_Level', 'Attrition_Flag']).sum().plot(kind='pie', subplots=True, shadow = True,startangle=90,figsize=(15,15), autopct='%1.1f%%')

sns.catplot(x="Education_Level", y="total",\
                hue="Attrition_Flag",\
                data=dfClientsPerdusNet.groupby(['Education_Level', 'Attrition_Flag'], as_index=False).agg(total=('Education_Level', 'count')), kind="bar",\
                height=4, aspect=3.0);
