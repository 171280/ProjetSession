# Importation des differentes librairies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import neighbors, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pickle
import os
import joblib
import json

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

model = DecisionTreeClassifier(max_depth = 13, random_state = 1)
model.fit(x_train, y_train)

prediction = model.predict(x_test)
print("La précision de l'arbre est de: ", "{:.3f}".format(metrics.accuracy_score(prediction, y_test)))

# confusion matrix
cf = metrics.confusion_matrix(y_test, prediction)
pd.DataFrame(cf, columns=['pred neg','pred pos'], index=['actual neg','actual pos'])

metrics.plot_confusion_matrix(model, x_test, y_test, display_labels=cn, cmap=plt.cm.Blues, normalize=None)

print('True Positive Rate/Recall = TP/(TP+FN) :', cf[1][1] / (cf[1][1] + cf[1][0])) 
print('False Positive Rate = FP/(FP+TN) :', cf[0][1] / (cf[0][1] + cf[0][0]))
print('Accuracy = (TP+TN)/total :', (cf[1][1]+cf[0][0])/(cf[1][1]+cf[0][0]+cf[0][1]+cf[1][0]))
print('Precision = TP/(TP+FP) :',  (cf[1][1]/(cf[1][1]+cf[0][1])))

# Model: Le plus proche voisin KNN
print("Model: Le plus proche voisin KNN")

# Créer l'objet Neighbours Classifier
# On considère KNN avec pondération uniforme
weights = 'uniform'
clf = neighbors.KNeighborsClassifier(weights = weights)
# Faire apprendre le model en utilisant les données d'entraînement
clf.fit(x_train, y_train)
# Prédire la classe pour toutes les observations dans les données d'entraînement
z = clf.predict(x_train)
print(z.shape)
# Comparer les classe prédictes avec les vrais labels de la classe
accuracy = clf.score(x_train, y_train)
print(f"La précision du model avec les données d'entraînement est de {accuracy}")

# Prédire la classe pour toutes les observations dans les données de test
z = clf.predict(x_test)
print(z.shape)
# Comparer les classe prédictes avec les vrais labels de la classe
accuracy = clf.score(x_test, y_test)
print(f"La précision du modèle avec les données de tests est de {accuracy}")

metrics.plot_confusion_matrix(clf, x_test, y_test, display_labels=cn, cmap=plt.cm.Blues, normalize=None)

## Naive Bayes
#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(x_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

metrics.plot_confusion_matrix(gnb, x_test, y_test, display_labels=cn, cmap=plt.cm.Blues, normalize=None)

# Regression Logistique
# set the model
logreg = LogisticRegression(C=3792.690190732246, penalty='l1', solver='liblinear')
# fit model
logreg.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = logreg.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

metrics.plot_confusion_matrix(logreg, x_test, y_test, display_labels=cn, cmap=plt.cm.Blues, normalize=None)

# Enregistrer le modèle
#pickle.dump(model, open('model.pkl', 'wb'))

os.makedirs("model_dir", exist_ok=True)
#model_path = os.path.join("model_dir", "model.joblib")
#joblib.dump(model, model_path)

print("Listing des repertoires:",os.listdir(os.getcwd()))
#os.chdir(os.path.join(os.sep,os.getcwd(),'model_dir'))
os.chdir('model_dir')
pickle.dump(model, open('model.pkl', 'wb'))
print("Listing des repertoires:",os.listdir(os.getcwd()))

# Ajouter
#Information sur le répertoire courant
#print ("Repertoire courant:",os.getcwd())
#os.chdir(os.getcwd())
#joblib.dump(model, os.getcwd())
#print("Listing des repertoires:",os.listdir(os.getcwd()))
