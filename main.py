# -*- coding: utf-8 -*-
"""
Treinamento de uma árvore decisão sobre o dataset Wine

@author: Fernando Gabriel
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn import tree
import pydotplus
import os

# Read dataset into data frame
train_df = pd.read_csv("wine.csv")

# Define labels (output) as 'Type'
# 'Type' is the target, that we want to predict from the values of the other columns
classes = ["1", "2", "3"] # 1 or 2 or 3
labels = "Type"
y = train_df["Type"].values

# Columns to classification
columns = ["Alcohol", 
           "Malicacid", 
           "Ash", 
           "Alcalinityofash", 
           "Magnesium", 
           "Total phenols", 
           "Flavanoids", 
           "Nonflavanoidphenols", 
           "Proanthocyanins", 
           "Colorintensity", 
           "Hue", 
           "OD280andOD315ofdilutedwines", 
           "Proline"]

features = train_df[list(columns)].values

# Replace 'nans' by mean to avoid issues
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(features)

# Learn the decision tree
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=13)
clf = clf.fit(X, y)

##Lembre-se de configurar o caminho do graphviz abaixo
os.environ["PATH"] += os.pathsep + 'C:\\Users\\FernandoGabriel\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\site-packages\\graphviz'  

# Export as png or pdf
#dot_data = tree.export_graphviz(clf, out_file=None, feature_names=columns) 
dot_data = tree.export_graphviz(clf, out_file=None,  feature_names=columns, 
                                class_names=classes,  filled=True,
                                rounded=True, special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_pdf("titanic.pdf")
graph.write_png('wine.png')


# Predict using our model

#14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065   Wine Type 1
 
#12.37,0.94,1.36,10.6,88,1.98,0.57,0.28,0.42,1.95,1.05,1.82,520    Wine Type 2

#12.86,1.35,2.32,18,122,1.51,1.25,0.21,0.94,4.1,0.76,1.29,630      Wine Type 3

Alcohol = 12.86
Malicacid = 1.35
Ash = 2.32
Alcalinityofash = 18
Magnesium = 122
Totalphenols = 1.51
Flavanoids = 1.25
Nonflavanoidphenols = 0.21
Proanthocyanins = 0.94
Colorintensity = 4.1
Hue = 0.76
OD280andOD315ofdilutedwines = 1.29
Proline = 630

# Predict a single decision, type 1, 2 or 3?
print(clf.predict([[Alcohol,
                    Malicacid, 
                    Ash, 
                    Alcalinityofash, 
                    Magnesium, 
                    Totalphenols, 
                    Flavanoids, 
                    Nonflavanoidphenols, 
                    Proanthocyanins, 
                    Colorintensity,
                    Hue,
                    OD280andOD315ofdilutedwines,
                    Proline]]))

# Predict probability of decision per class
print(clf.predict_proba([[Alcohol,
                          Malicacid, 
                          Ash, 
                          Alcalinityofash, 
                          Magnesium, 
                          Totalphenols, 
                          Flavanoids, 
                          Nonflavanoidphenols, 
                          Proanthocyanins, 
                          Colorintensity,
                          Hue,
                          OD280andOD315ofdilutedwines,
                          Proline]]))