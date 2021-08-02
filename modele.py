# Importation des differentes librairies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from   sklearn import neighbors, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pickle

print("Nous allons developper des modèles")
print("Nous allons diviser les données en deux parties. Une partie pour l'entrainement et l'autre partie pour le test.")

dfClientsPerdusNet = pd.read_csv("BankChurners_aed_pretrait.csv")

train, test = train_test_split(dfClientsPerdusNet, test_size = 0.2, stratify=dfClientsPerdusNet['Attrition_Flag'], random_state = 44)

x_train = train[['Customer_Age', 'Dependent_count', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', \
                 'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', \
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',\
                 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]
y_train = train.Attrition_Flag
x_test  = test[['Customer_Age', 'Dependent_count', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', \
                 'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', \
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',\
                 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]
y_test  = test.Attrition_Flag
fn = ['Customer_Age', 'Dependent_count', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', \
                 'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', \
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',\
                 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
cn= ['0', '1']

#detail de chacun des sous-dataset
print(dfClientsPerdusNet.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print("Model: Arbre de décision")
