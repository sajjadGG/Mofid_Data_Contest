import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df.head()

df['Text'] = df['Text'].fillna("NULL!!!!")

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(df['Text'])

from sklearn.model_selection import train_test_split
X , y = X , df['Class']
Xtrain , Xtest , ytrain , ytest = train_test_split(X,y,test_size=0.25)


from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import GradientBoostingClassifier
# model = LogisticRegression()
model.fit(Xtrain , ytrain)

print(model.score(Xtest , ytest))

from sklearn.pipeline import Pipeline
pipe = Pipeline([('tfidf', vectorizer), ('model',model)])

dftest = pd.read_csv('test.csv')

predictions = pipe.predict_proba(dftest['Text'])

pd.DataFrame(predictions[:,1] , columns=['pred']).to_csv('pred.csv' , index=False)