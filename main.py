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
#graph.write_pdf("wine.pdf")
graph.write_png('wine.png')


# Predict using our model

# Wine Type 1:
#Alcohol                     = 14.23
#Malicacid                   = 1.71
#Ash                         = 2.43
#Alcalinityofash             = 15.6
#Magnesium                   = 127
#Totalphenols                = 2.8
#Flavanoids                  = 3.06
#Nonflavanoidphenols         = 0.28
#Proanthocyanins             = 2.29
#Colorintensity              = 5.64
#Hue                         = 1.04
#OD280andOD315ofdilutedwines = 3.92
#Proline                     = 1065

# Wine Type 2:
#Alcohol                     = 12.37
#Malicacid                   = 0.94
#Ash                         = 1.36
#Alcalinityofash             = 10.6
#Magnesium                   = 88
#Totalphenols                = 1.98
#Flavanoids                  = 0.57
#Nonflavanoidphenols         = 0.28
#Proanthocyanins             = 0.42
#Colorintensity              = 1.95
#Hue                         = 1.05
#OD280andOD315ofdilutedwines = 1.82
#Proline                     = 520

# Wine Type 3:
Alcohol                     = 12.86
Malicacid                   = 1.35
Ash                         = 2.32
Alcalinityofash             = 18
Magnesium                   = 122
Totalphenols                = 1.51
Flavanoids                  = 1.25
Nonflavanoidphenols         = 0.21
Proanthocyanins             = 0.94
Colorintensity              = 4.1
Hue                         = 0.76
OD280andOD315ofdilutedwines = 1.29
Proline                     = 630

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