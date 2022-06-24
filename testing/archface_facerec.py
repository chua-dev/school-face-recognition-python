import cv2
import time
import numpy as np
import mediapipe as mp
from arcface import ArcFace
from imutils.video import WebcamVideoStream
import os

# Load Arc Face Model
face_rec = ArcFace.ArcFace()

# Load Detection Model
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.45,1)

#camera_id = 0
camera_id = "rtsp://admin:abc12345@192.168.1.5:554/Stream/Channels/101"
cap = WebcamVideoStream(src=camera_id).start()

#emb1 = face_rec.calc_emb('chua.jpeg')
#emb2 = face_rec.calc_emb('chua2.jpeg')
#emb3 = face_rec.calc_emb('jack.jpeg')

#path = 'C:/Users/user/Desktop/school-multiprocessing-facerec/face'
#path = "/Users/p2digital/Documents/school-multiprocessing-facerec/testing/face"
path = "/home/p2d/Documents/CCTV_FACE_PROJECT/school-face-recognition-python/face"

name_dict=list()
FULL_REGISTER_LIST=list()
i = 0
for folders in os.listdir(path):
    try:
        #name_dict[i] = folders
        truePath = os.path.join(path,folders)
        for image in os.listdir(truePath):
            img = cv2.imread(os.path.join(truePath,image))
            height, width, channels = img.shape
            #print(height, width, channels)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            detections = faceDetection.process(img)
            r_bounding_box = detections.detections[0].location_data.relative_bounding_box
            
            x, y, h, w = round(r_bounding_box.xmin * width) + 0, round(r_bounding_box.ymin * height) + 0, round(r_bounding_box.height * height) + 0, round(r_bounding_box.width * width) + 0

            img = img[y:y+h, x:x+w]
            name_dict.append(folders)

            each_emb = face_rec.calc_emb(img)
            FULL_REGISTER_LIST.append(each_emb)
            #cv2.imshow('image',img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #img = image_resize(img, height = 140)
            #x_list.append(imgEncode)
            #photo_list.append(img)
            #y_list.append(i)
        i += 1
    except:
        print('unbuildable folder')

#emb_list = [emb1, emb2]
emb_list = FULL_REGISTER_LIST

print(name_dict)
while True:
    frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #width, height = frame.shape[:2]
    height, width, channels = frame.shape
    #print(width, height, 1280, 720, 16:9)

    detections = faceDetection.process(rgb_frame)
    if detections.detections:
        #print(detections.detections[0].location_data.relative_bounding_box)
        r_bounding_box = detections.detections[0].location_data.relative_bounding_box
        x, y, h, w = round(r_bounding_box.xmin * width) + 0, round(r_bounding_box.ymin * height) + 0, round(r_bounding_box.height * height) + 0, round(r_bounding_box.width * width) + 0

        crop_frame = rgb_frame[y:y+h, x:x+w]
        crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_RGB2BGR)

        # Actual Calculation
        emb = face_rec.calc_emb(crop_frame)
        #emb = face_rec.calc_emb(crop_frame)
        distance_list = list()

        for each_emb in emb_list:
            distance = face_rec.get_distance_embeddings(emb, each_emb)
            distance_list.append(distance)
    
        #print(distance_list)
        idx = np.argmin(distance_list)
        value = np. amin(distance_list)
        print(f"{name_dict[idx]}, {value}")
        if name_dict[idx] != 'Chua_30038':
            print(distance_list)
        cv2.imshow("webcam", crop_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow('Image')




'''
emb1 = face_rec.calc_emb("jack.jpeg")
emb2 = face_rec.calc_emb("john.jpeg")
emb3 = face_rec.calc_emb("jenny.jpeg")

distance = face_rec.get_distance_embeddings(emb2, emb3)
'''