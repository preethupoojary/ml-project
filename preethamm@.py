# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 21:48:26 2023

@author: PREETHAM POOJARY
"""
#first question

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
names = ['id', 'clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion',
         'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
data = pd.read_csv(url, names=names)
data = data.drop('id', axis=1) 

data = data.replace({'?':np.nan})
data = data.dropna()  
data['class'] = data['class'].replace(2, 0) 
data['class'] = data['class'].replace(4, 1)  

X = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)
 
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)

results = pd.DataFrame({'Method': ['Logistic Regression', 'KNN', 'Naive Bayes'],
                        'Accuracy': [acc_lr, acc_knn, acc_nb],
                        'Confusion Matrix': [cm_lr, cm_knn, cm_nb]})
print(results)




#sec question


import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

url = 'https://monkeylearn.com/sentiment-analysis/'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
text = soup.find_all('p')

content = ''
for p in text:
    content += p.get_text()

sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores(content)

if sentiment['compound'] > 0:
    print('Positive')
elif sentiment['compound'] < 0:
    print('Negative')
else:
    print('Neutral')
    
    
    
#third question


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv(r"C:\Users\HP\AppData\Local\Packages\microsoft.windowscommunicationsapps_8wekyb3d8bbwe\LocalState\Files\S0\5\Attachments\CC GENERAL[39].csv")

data = data.drop(['CUST_ID'], axis=1)

data = data.fillna(data.mean())

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data_pca)
clusters = kmeans.predict(data_pca)

plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Credit Card Users Clusters')
plt.show()















