"""
For model training, testing and validation
"""

from face_recognition import EigenFace
#############3
#import pandas as pd
#import numpy as np
#train_test_split
#pd.DataFrame
#pd.read_csv
#np.array
#e = EigenFace(10,100)
#e.readLabels
####################
from sklearn.model_selection import train_test_split

datasetPath = "E:/Machine Learning/Datasets/bct"
fit_mode = 'train'


eigenface = EigenFace(image_x=200, image_y=200)
df = eigenface.generateLabels(dataset_path=datasetPath)
print('Step 1 done')
X,y = eigenface.readLabels(label_df=df) 
print('Step 2 done')

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.5,test_size=0.5,stratify=y)
print('Step 3 done')
eigenface.saveModel(label_df = df)
eigenface.fit(X_train,y_train,mode=fit_mode)
print('Step 4 done')

y_pred = eigenface.predict(X_test,y_test)

#%%

#analysis of the program
y_comp=[]
#y_comp = ['True' for i in range(len(y_pred)) if y_pred[i]==y_test[i] else 'False']
for i in range(len(y_pred)):
    if y_pred[i]==y_test[i]:
        y_comp.append('True')
    else:
        y_comp.append('False')
y_base = ['True']*len(y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_comp, y_base)