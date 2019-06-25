from face_recognition import EigenFace
import sys

datasetPath = "E:/Machine Learning/Datasets/bct"
img_x = 200
img_y = 200

# datasetPath = sys.argv[1]
# img_x = sys.argv[2]
# img_y = sys.argv[3]

eigenface = EigenFace(image_x=img_x, image_y=img_y)
df = eigenface.generateLabels(dataset_path=datasetPath)
X,y = eigenface.readLabels(label_df=df) 

eigenface.fit(X,y,mode='train')
eigenface.saveModel(label_df = df)