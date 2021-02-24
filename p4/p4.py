import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fill(df):
    
    dfresult = pd.concat([df[bincolumns[:3]].fillna(2),
    df[bincolumns[3:]].fillna('U'),
    df[normcolumns].fillna("UNKNOWN!!!"),
    df[ordcolumns[0]].fillna(0) ,
    df[ordcolumns[1:]].fillna("UNKNOWN"),
    df['day'].fillna(0),
    df['month'].fillna(0)] , axis=1)
    ord1 = {'UNKNOWN':0 , 'Novice':1 ,  'Contributor':2 , 'Expert':3 , 'Master' : 4 , 'Grandmaster':5}
    ord2 = {'UNKNOWN':0 , 'Cold':-1 ,  'Freezing':-2 , 'Warm':1 , 'Hot' : 2 , 'Boiling Hot':3 , 'Lava Hot':4}
    ord3 = {chr(k):k for k in range(ord('a') , ord('z')+1) }
    ord3['UNKNOWN']=0
    ord4 = {chr(k):k for k in range(ord('A') , ord('Z')+1) }
    ord4['UNKNOWN']=0


    dfresult['ord_1'] = dfresult['ord_1'].apply(lambda x : ord1[x])
    dfresult['ord_2'] = dfresult['ord_2'].apply(lambda x : ord2[x])
    dfresult['ord_3'] = dfresult['ord_3'].apply(lambda x : ord3[x])
    dfresult['ord_4'] = dfresult['ord_4'].apply(lambda x : ord4[x])
    return dfresult
#result = pd.concat([df1, df4], axis=1)

df = pd.read_csv('train.csv')
columns = df.columns
bincolumns = [x for x in columns[:5]]
normcolumns = [x for x in columns[5:15]]
ordcolumns = [x for x in columns[15:20]]

dfresult = pd.concat([df[bincolumns[:3]].fillna(2),
df[bincolumns[3:]].fillna('U'),
df[normcolumns].fillna("UNKNOWN!!!"),
df[ordcolumns[0]].fillna(0) ,
df[ordcolumns[1:]].fillna("UNKNOWN"),
df['day'].fillna(0),
df['month'].fillna(0)] , axis=1)

ord1 = {'UNKNOWN':0 , 'Novice':1 ,  'Contributor':2 , 'Expert':3 , 'Master' : 4 , 'Grandmaster':5}
ord2 = {'UNKNOWN':0 , 'Cold':-1 ,  'Freezing':-2 , 'Warm':1 , 'Hot' : 2 , 'Boiling Hot':3 , 'Lava Hot':4}
ord3 = {chr(k):k for k in range(ord('a') , ord('z')+1) }
ord3['UNKNOWN']=0
ord4 = {chr(k):k for k in range(ord('A') , ord('Z')+1) }
ord4['UNKNOWN']=0


dfresult['ord_1'] = dfresult['ord_1'].apply(lambda x : ord1[x])
dfresult['ord_2'] = dfresult['ord_2'].apply(lambda x : ord2[x])
dfresult['ord_3'] = dfresult['ord_3'].apply(lambda x : ord3[x])
dfresult['ord_4'] = dfresult['ord_4'].apply(lambda x : ord4[x])

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
enc = OneHotEncoder(handle_unknown='ignore' )
enc.fit(dfresult[bincolumns + normcolumns])
# svd = TruncatedSVD(n_components=32, n_iter=7, random_state=42)
X = enc.transform(dfresult[bincolumns + normcolumns])
# Xtrans = svd.fit_transform(X)

from scipy.sparse import hstack
X=hstack((X, dfresult[ordcolumns + ['day' , 'month']].values))
# X = np.concatenate((Xtrans , dfresult[ordcolumns + ['day' , 'month']].values) , axis=1)

from sklearn.model_selection import train_test_split
X , y = X , df['target']
Xtrain , Xtest , ytrain , ytest = train_test_split(X,y,test_size=0.25)

from sklearn.linear_model import LogisticRegression  , LinearRegression
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier , AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier , ExtraTreeClassifier

# model = LinearRegression( )
# model = GradientBoostingClassifier()
model = RandomForestClassifier(n_estimators=200 )
# model = ExtraTreeClassifier()
model.fit(Xtrain , ytrain)


print(model.score(Xtrain , ytrain) , model.score(Xtest , ytest))

dftest = pd.read_csv('test.csv')
dftest = fill(dftest)

Xt = enc.transform(dftest[bincolumns + normcolumns])
# Xt = svd.transform(Xt)

Xt=hstack((Xt, dftest[ordcolumns + ['day' , 'month']].values))
# Xt = np.concatenate((Xt , dftest[ordcolumns + ['day' , 'month']].values) , axis=1)

predictions = model.predict_proba(Xt)

pd.DataFrame(predictions[:,1] , columns=['pred']).to_csv('pred.csv' , index=False)

