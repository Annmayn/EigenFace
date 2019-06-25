import numpy as np 
import cv2
from os import listdir, mkdir
from os.path import join, isfile
import os

root_path = "E:/Machine Learning/Datasets/"
dataset_name = "bct1/"
dataset_path = root_path+dataset_name
face_classifier = cv2.CascadeClassifier("etc/haarcascade_frontalface_default.xml")

def face_extractor(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.1,0)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_faces = gray[y:y+h, x:x+w]
    return cropped_faces

cnt=0
for img_name in listdir(dataset_path):
    img_dir = dataset_path + img_name + '/'
    only_images = [f for f in listdir(img_dir) if isfile(join(img_dir,f))]

    for img in only_images:
        img_path = img_dir+img
        # print(img_path)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Section that removes photos where the algorithm can't detect face

        cropped_img = face_extractor(img_path) 
        if cropped_img is not None:
            cropped_img = cv2.resize(cropped_img, (200,200))
            cnt+=1
            # print(img_path)
            cv2.imwrite(img_path, cropped_img)
        else:
            print(img_path)
            os.remove(img_path)


