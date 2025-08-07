#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

dataFrame = pd.read_csv('spotify_history.csv')

dataFrame = dataFrame[dataFrame['ms_played'] >= 30000]

songPlayCounts = dataFrame.groupby('spotify_track_uri').size().reset_index(name='playCount')
dataFrame = dataFrame.merge(songPlayCounts, on='spotify_track_uri')
dataFrame['replayedWithinMonth'] = (dataFrame['playCount'] > 1).astype(int)

dataFrame['shuffle'] = dataFrame['shuffle'].astype(int)
dataFrame['skipped'] = dataFrame['skipped'].astype(int)

featureList = ['ms_played', 'shuffle', 'skipped']
X = dataFrame[featureList]
y = dataFrame['replayedWithinMonth']

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(XTrain, yTrain)

yPred = model.predict(XTest)
print("Classification Report:")
print(classification_report(yTest, yPred))

sns.set(style="whitegrid")

plt.figure(figsize=(10, 4))
sns.histplot(dataFrame['ms_played'] / 1000, bins=50, kde=True)
plt.title('Distribution of Play Duration in seconds')
plt.xlabel('Play Duration in seconds')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='skipped', data=dataFrame)
plt.title('Skipped vs Not Skipped')
plt.xticks([0, 1], ['Not Skipped', 'Skipped'])
plt.ylabel('Number of Plays')
plt.show()

importances = model.feature_importances_
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=featureList)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x=model.predict(X), palette="viridis")
plt.title('Replay Prediction Count')
plt.xticks([0, 1], ['Not Replayed', 'Replayed'])
plt.ylabel('Number of Predictions')
plt.show()

plt.figure(figsize=(6, 5))
correlationMatrix = dataFrame[featureList + ['replayedWithinMonth']].corr()
sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:




