from face_recognition import EigenFace
import cv2
import datetime
import time
import json
import os
import sys
from os import listdir
from os.path import join, isfile, exists
face_classifier = cv2.CascadeClassifier("etc/haarcascade_frontalface_default.xml")
#Video_filename = "attendance_video.avi"

#runtime value from console
runtime = int(sys.argv[1])

#Threshold value for image prediction
#lower => more accurate but rejects more image 
#higher => less accurate but recognizes more image
thd_value = 1400000

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=5, minSize=(20,20))
    
    if faces is():
        return img, []
    
    return gray, faces

cap = cv2.VideoCapture(0)
model = EigenFace()
model.loadModel()

attendance_date = datetime.datetime.now().strftime("%Y-%m-%d")

predictedNames = {}
t1 = time.time()
t2 = time.time()


#Create missing directories
if not os.path.exists("tmp"):
    os.makedirs("tmp")
if not os.path.exists("tmp/"+attendance_date):
    os.makedirs("tmp/"+attendance_date)
    
while t2-t1<runtime:
    ret, frame = cap.read()
    if ret == True:
        gray, faces = face_detector(frame)

        if len(faces)!=0:
            for (x,y,w,h) in faces:
                face_save =  gray[y:y+h, x:x+w]
                face = cv2.resize(face_save,(200,200))
                # face_save = cv2.GaussianBlur(face_save, (5,5), 0)
                if len(face)!=0:
                    y_pred,comp = model.image_predict(face, threshold=thd_value)
                    
                    #skip if no name is returned
                    if y_pred is None:
                        continue

                    if not os.path.exists("tmp/"+attendance_date+"/"+y_pred):
                        os.makedirs("tmp/"+attendance_date+"/"+y_pred)
                    #set to 1 if not already present otherwise increase count
                    try:
                        predictedNames[y_pred]+=1
                    except:
                        predictedNames[y_pred]=1
                    path_to_image = join('tmp',attendance_date,y_pred,datetime.datetime.now().strftime("%H-%M-%S-%f")+'.png')
                    
                    # print(path_to_image)
                    if predictedNames[y_pred]<=10 or True:
                        cv2.imwrite(path_to_image,face_save)
                        # cv2.imwrite(join('tmp',attendance_date,y_pred,datetime.datetime.now().strftime("%H-%M-%S-%f")+'prediction.png'), cv2.imread(comp))
            
        if cv2.waitKey(1)==27:
            break
    else:
        break

    t2 = time.time()
# with open('attendance.json', 'w+') as f:
    # f.write(str(saveFile))
cap.release()
cv2.destroyAllWindows()

#CREATE JSON OF THE ATTENDANCE

if not os.path.exists("tmp/json"):
    os.makedirs("tmp/json")

json_list = []
load_root = join(os.getcwd(), "tmp")
save_root = join(load_root, "json")

for num,dirr in enumerate(listdir(join(load_root, attendance_date))):
    user_json = {}
    user_json['name'] = dirr
    user_json['status'] = 'P'
    
    tmp_dict = {}
    data_path = join(load_root, attendance_date, dirr)
    only_images = [f for f in listdir(data_path) if isfile(join(data_path,f))]
  
    for i,image_name in enumerate(only_images):
        tmp_dict['url'+str(i+1)] = join(data_path, image_name)
    user_json['url'] = tmp_dict
    json_list.append(user_json)

with open(join(save_root,attendance_date)+".json", "w+") as f:
    f.write(str(json_list)) 