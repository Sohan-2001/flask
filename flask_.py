from flask import Flask, request
import cv2
import dlib
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)   # Flask constructor 
  
# A decorator used to tell the application 
# which URL is associated function 
@app.route('/')       
def hello(): 
    imagePath = request.args.get('imagePath')
    if imagePath is None:
        return 'No image path provided'

    response = requests.get(imagePath)
    if response.status_code != 200:
        return 'Image could not be retrieved'

    image_bytes = BytesIO(response.content)
    img = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (500, 700))
    frame = img.copy()

    # ------------ Model for Age detection --------# 
    age_weights = "age_deploy.prototxt"
    age_config = "age_net.caffemodel"
    age_Net = cv2.dnn.readNet(age_config, age_weights) 

    # Model requirements for image 
    ageList = ['(0-5)', '(5-10)', '(10-15)', '(15-24)', '(25-34)', '(35-44)', '(45-54)', '(55-64)', '(65-74)','(75-84)','(85-94)','(95-105)']
    model_mean = (78.4263377603, 87.7689143744, 114.895847746) 

    # storing the image dimensions 
    fH = img.shape[0] 
    fW = img.shape[1] 
    Boxes=[]

    #------------- Model for face detection---------# 
    face_detector = dlib.get_frontal_face_detector() 
    # converting to grayscale 
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector(img_gray)
    
    if not faces:
        
        return 'No Face Detected'
    
    else: 
	    # --------- Bounding Face ---------# 
        for face in faces: 
            x = face.left() # extracting the face coordinates 
            y = face.top() 
            x2 = face.right() 
            y2 = face.bottom() 

		    # rescaling those coordinates for our image 
            box = [x, y, x2, y2] 
            Boxes.append(box) 
            cv2.rectangle(frame, (x, y), (x2, y2), 
					(00, 200, 200), 2) 

        for box in Boxes: 
            face = frame[box[1]:box[3], box[0]:box[2]] 

		    # ----- Image preprocessing --------# 
            blob = cv2.dnn.blobFromImage( 
			    face, 1.0, (227, 227), model_mean, swapRB=False) 

		    # -------Age Prediction---------# 
            age_Net.setInput(blob) 
            age_preds = age_Net.forward() 
            age = ageList[age_preds[0].argmax()]

            return age
  
if __name__=='__main__':
   app.run()
