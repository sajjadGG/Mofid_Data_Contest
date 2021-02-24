import IPython.display as ipd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam


from os import listdir
from os.path import isfile, join
onlyfiles = [join(join(cwd, "train"),f) for f in listdir(cwd+'\\train') ]


l = []
for o in [f for f in listdir(cwd+'\\train') ]:
    l.append((o[4],o[5]))

df = pd.DataFrame.from_records(l , columns=['gender' , 'target'])
df.head()


mfc=[]
chrs=[]
me=[]
ton=[]
lab=[]
for i,e in enumerate(onlyfiles):
    f_name=e
    X, s_rate = librosa.load(f_name, res_type='kaiser_fast')
    mf = np.mean(librosa.feature.mfcc(y=X, sr=s_rate).T,axis=0)
    mfc.append(mf)
    l=df.iloc[i]['target']
    lab.append(l)
    try:
        t =  np.mean(librosa.feature.tonnetz(
                       y=librosa.effects.harmonic(X),
                       sr=s_rate).T,axis=0)
        ton.append(t)
    except:
        print(f_name)  
    m = np.mean(librosa.feature.melspectrogram(X, sr=s_rate).T,axis=0)
    me.append(m)
    s = np.abs(librosa.stft(X))
    c = np.mean(librosa.feature.chroma_stft(S=s, sr=s_rate).T,axis=0)
    chrs.append(c)


features = []
for i in range(len(ton)):
    features.append(np.concatenate((me[i], mfc[i], 
                ton[i], chrs[i]), axis=0))



la = pd.get_dummies(lab)
label_columns=la.columns #To get the classes
target = la.to_numpy() #Convert labels to numpy array




tran = StandardScaler()
features_train = tran.fit_transform(features)


nt = 1900
nv = 2100
feat_train=features_train[:nt]
target_train=target[:nt]
y_train=features_train[nt:nv]
y_val=target[nt:nv]
test_data=features_train[nv:]
test_label=target[nv:]
print("Training",feat_train.shape)
print(target_train.shape)
print("Validation",y_train.shape)
print(y_val.shape)
print("Test",test_data.shape)
print(test_label.shape)


from tensorflow.keras import regularizers
inp_shape = 166
out_shape=5
model = Sequential()

model.add(Dense(inp_shape, input_shape=(inp_shape,), activation = 'relu' , activity_regularizer=regularizers.l2(1e-3)))

model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.6))

model.add(Dense(16, activation = 'relu' , activity_regularizer=regularizers.l2(1e-3)))
model.add(Dropout(0.5))

model.add(Dense(8, activation = 'relu' , activity_regularizer=regularizers.l2(1e-3)))
model.add(Dropout(0.5))

model.add(Dense(out_shape, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(feat_train, target_train, batch_size=2048, epochs=200, 
                    validation_data=(y_train, y_val))



print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


from sklearn.linear_model import LogisticRegression  , LinearRegression
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier , AdaBoostClassifier , BaggingClassifier
from sklearn.tree import DecisionTreeClassifier , ExtraTreeClassifier
from sklearn import svm
model2 = svm.SVC(probability=True,degree=12,kernel='linear' , C=0.01)
# model2 = LogisticRegression( )
# model2 = AdaBoostClassifier()
# model2 = RandomForestClassifier(n_estimators=50)
# model = BaggingClassifier()
# model = ExtraTreeClassifier()
model2.fit(feat_train , lab[:nt])


model2.score(feat_train , lab[:nt]) , model2.score(y_train , lab[nt:nv]) 

weights = [795/2221 , 152/2221 , 782/2221 , 329/2221 , 163/2221]
sum(weights)

l=[]
maxs = [0]*5
for i in range(len(output)):
    tr=0
    
    a = np.random.dirichlet(weights,size=1)
#     while(np.argmax(a)!=0 or np.argmax(a)!=2) :
#         if tr>10:
#             break
#         a = np.random.dirichlet(np.array(weights),size=1)
#         tr+=1
    maxs[np.argmax(a)]+=1
#     l.append([int(output.iloc[i]['file_id'])] + [a[0][i] for i in range(5)])
    l.append([int(output.iloc[i]['file_id'])] + weights)




    def transform(name='test'):
    onlyfiles = [join(join(cwd, name),f) for f in listdir(cwd+'\\{}'.format(name)) ]
    mfc=[]
    chrs=[]
    me=[]
    ton=[]
    lab=[]
    for i,e in enumerate(onlyfiles):

        f_name=e
        X, s_rate = librosa.load(f_name, res_type='kaiser_fast')
        mf = np.mean(librosa.feature.mfcc(y=X, sr=s_rate).T,axis=0)
        mfc.append(mf)
        l=df.iloc[i]['target']
        lab.append(l)
        try:
            t =  np.mean(librosa.feature.tonnetz(
                           y=librosa.effects.harmonic(X),
                           sr=s_rate).T,axis=0)
            ton.append(t)
        except:
            print(f_name)  
        m = np.mean(librosa.feature.melspectrogram(X, sr=s_rate).T,axis=0)
        me.append(m)
        s = np.abs(librosa.stft(X))
        c = np.mean(librosa.feature.chroma_stft(S=s, sr=s_rate).T,axis=0)
        chrs.append(c)
        
    tran = StandardScaler()
    features_train = tran.fit_transform(features)
    return feat_train




tesfeat = transform()




predict = model2.predict_proba(tesfeat)




l=[]
for i in range(len(output)):

    a = predict[i]
    #print(a)
    l.append([int(output.iloc[i]['file_id'])] + list(a))


dfoutput = pd.DataFrame.from_records(l , columns=output.columns)


dfoutput.to_csv('outputpred.csv' , index=False)