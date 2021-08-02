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

def with_hue(plot, feature, Number_of_categories, hue_categories):
    a = [p.get_height() for p in plot.patches]
    patch = [p for p in plot.patches]
    for i in range(Number_of_categories):
        total = feature.value_counts().values[i]
        for j in range(hue_categories):
            percentage = '{:.1f}%'.format(100 * a[(j*Number_of_categories + i)]/total)
            x = patch[(j*Number_of_categories + i)].get_x() + patch[(j*Number_of_categories + i)].get_width() / 2 - 0.15
            y = patch[(j*Number_of_categories + i)].get_y() + patch[(j*Number_of_categories + i)].get_height() 
            ax.annotate(percentage, (x, y), size = 12)
    #plt.show()

def without_hue(plot, feature):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), size = 12)
    #plt.show()

plt.figure(figsize=(15,5))
ax=sns.countplot('Education_Level', hue='Attrition_Flag',data=dfClientsPerdusNet)
with_hue(ax, dfClientsPerdusNet.Education_Level, 2, 7)

plt.figure(figsize=(15,5))
ax=sns.countplot('Education_Level',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Education_Level)

plt.figure(figsize=(15,5))
ax=sns.countplot('Education_Level', hue='Attrition_Flag',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Education_Level)

# Analyse de la variable "Education_Level"
plt.figure(figsize=(15,10))
plt.subplot(2,1,1)
ax=sns.countplot('Education_Level',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Education_Level)
plt.subplot(2,1,2)
ax=sns.countplot('Education_Level', hue='Attrition_Flag',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Education_Level)
plt.show()

# Analyse de la variable "Marital_Status"
plt.figure(figsize=(15,10))
plt.subplot(2,1,1)
ax=sns.countplot('Marital_Status',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Marital_Status)
plt.subplot(2,1,2)
ax=sns.countplot('Marital_Status', hue='Attrition_Flag',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Marital_Status)
plt.show()

# Analyse de la variable "Income_Category"
plt.figure(figsize=(15,10))
plt.subplot(2,1,1)
ax=sns.countplot('Income_Category',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Income_Category)
plt.subplot(2,1,2)
ax=sns.countplot('Income_Category', hue='Attrition_Flag',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Income_Category)
plt.show()

# Analyse de la variable "Card_Category"
plt.figure(figsize=(15,10))
plt.subplot(2,1,1)
ax=sns.countplot('Card_Category',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Card_Category)
plt.subplot(2,1,2)
ax=sns.countplot('Card_Category', hue='Attrition_Flag',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Card_Category)
plt.show()

# Analyse de la variable "Gender"
plt.figure(figsize=(15,10))
plt.subplot(2,1,1)
ax=sns.countplot('Gender',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Gender)
plt.subplot(2,1,2)
ax=sns.countplot('Gender', hue='Attrition_Flag',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Gender)
plt.show()

# Analyse de la variable "Attrition_Flag"
plt.figure(figsize=(15,10))
plt.subplot(2,1,1)
ax=sns.countplot('Attrition_Flag',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Attrition_Flag)
plt.subplot(2,1,2)
ax=sns.countplot('Attrition_Flag', hue='Attrition_Flag',data=dfClientsPerdusNet)
without_hue(ax, dfClientsPerdusNet.Attrition_Flag)
plt.show()

dfClientsPerdusNet.plot(kind='box', figsize = (15,10), rot=30, showfliers=False, vert=False);

sns.pairplot(dfClientsPerdusNet, hue='Attrition_Flag')

sns.pairplot(dfClientsPerdusNet, vars=['Total_Trans_Amt','Total_Trans_Ct'], hue='Attrition_Flag')

# Vérifions la correlation
fig = plt.subplots(figsize=(15, 10))
matrice = dfClientsPerdusNet.corr().round(2)
sns.heatmap(data=matrice, annot=True)
plt.show()

# Nous allons faire une pré-traitement ou une transformation de nos données

dfClientsPerdusNet['Income_Category'].value_counts()
dfClientsPerdusNet['Attrition_Flag'].value_counts()
dfClientsPerdusNet['Marital_Status'].value_counts()
dfClientsPerdusNet['Card_Category'].value_counts()
dfClientsPerdusNet['Education_Level'].value_counts()
dfClientsPerdusNet['Gender'].value_counts()

dfClientsPerdusNet[list(set(dfClientsPerdusNet.columns) - set(dfClientsPerdusNet._get_numeric_data().columns))]

dfClientsPerdusNet[list(set(dfClientsPerdusNet.columns) - set(dfClientsPerdusNet._get_numeric_data().columns))].loc[:,:]

le=LabelEncoder()
for i in dfClientsPerdusNet[list(set(dfClientsPerdusNet.columns) - set(dfClientsPerdusNet._get_numeric_data().columns))].loc[:,:]:
   dfClientsPerdusNet[i]=le.fit_transform(dfClientsPerdusNet[i])
   
# Vérifions la correlation
fig = plt.subplots(figsize=(15, 10))
matrice = dfClientsPerdusNet.corr().round(2)
sns.heatmap(data=matrice, annot=True)
plt.show()

#sauvegarder le data
dfClientsPerdusNet.to_csv("BankChurners_aed_pretrait.csv",index=False)
