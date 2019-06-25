from face_recognition import EigenFace
import cv2
face_classifier = cv2.CascadeClassifier("etc/haarcascade_frontalface_default.xml")

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    
    if faces is():
        return img, []
    
    for (x,y,w,h) in faces:
#        cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,255), 2)
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img, roi

cap = cv2.VideoCapture(0)
model = EigenFace()
model.loadModel()

saveFile = {}

counter = 0
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    if len(face)!=0:
        counter+=1
        y_pred = model.image_predict(face)
        print(y_pred)
        saveFile[str(counter)+'_name']=y_pred
        saveFile[str(counter)+'_image']=image
        
#        cv2.putText(image, 'hi', (200,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),1)
#        cv2.imshow('Face Recognizer', image)
    # else:
    #     cv2.putText(image, "Not found", (200,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),1)
    #     cv2.imshow('Face Recognizer', image)
    if cv2.waitKey(1)==27:
        break
    
    with open('attendance.json', 'w+') as f:
        f.write(str(saveFile))