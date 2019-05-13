from FaceRecognition import EigenFace
from sklearn.model_selection import train_test_split

datasetPath = "E:/Machine Learning/Datasets/orl_faces original"
fit_mode = 'train'


eigenface = EigenFace(image_x=112, image_y=92)
df = eigenface.generateLabels(dataset_path=datasetPath)
print('Step 1 done')
X,y = eigenface.readLabels(label_df=df) 
print('Step 2 done')

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.4,test_size=0.6,stratify=y)
print('Step 3 done')
eigenface.saveModel(label_df = df)
eigenface.fit(X_train,y_train,mode=fit_mode)
print('Step 4 done')

y_pred = eigenface.predict(X_test,y_test)
