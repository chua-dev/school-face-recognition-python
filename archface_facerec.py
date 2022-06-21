import cv2
import time
import numpy as np
import mediapipe as mp
from arcface import ArcFace
from imutils.video import WebcamVideoStream

# Load Arc Face Model
face_rec = ArcFace.ArcFace()

# Load Detection Model
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.45,1)

camera_id = 0
cap = WebcamVideoStream(src=camera_id).start()
emb1 = face_rec.calc_emb('chua.jpeg')
emb2 = face_rec.calc_emb('chua2.jpeg')
emb3 = face_rec.calc_emb('jack.jpeg')

#emb_list = [emb1, emb2]
emb_list = [emb1, emb2, emb3]

while True:
    frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #width, height = frame.shape[:2]
    #print(width, height, 1280, 720, 16:9)
    
    resized_frame = cv2.resize(rgb_frame, (640, 360))
    detections = faceDetection.process(resized_frame)
    if detections.detections:
        #print(detections.detections[0].location_data.relative_bounding_box)
        r_bounding_box = detections.detections[0].location_data.relative_bounding_box
        x, y, h, w = round(r_bounding_box.xmin * 640) + 0, round(r_bounding_box.ymin * 360) + 0, round(r_bounding_box.height * 360) + 0, round(r_bounding_box.width * 640) + 0

        crop_frame = resized_frame[y:y+h, x:x+w]
        crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_RGB2BGR)

        # Actual Calculation
        emb = face_rec.calc_emb(crop_frame)
        #emb = face_rec.calc_emb(crop_frame)
        distance_list = list()

        for each_emb in emb_list:
            distance = face_rec.get_distance_embeddings(emb, each_emb)
            distance_list.append(distance)
    
        lowest_value_index = np.argmin(distance)
        print(lowest_value_index)
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