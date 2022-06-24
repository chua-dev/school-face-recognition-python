import os
import cv2
import time
import datetime
import logging
import logging.handlers as handlers
import numpy as np
import mediapipe as mp
from arcface import ArcFace
from imutils.video import WebcamVideoStream

from utils import *

logging.basicConfig(filename="./log/attendance.log", format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.debug("Initiating the Attendance System")

# Load Arc Face Model
logger.debug("Loading Face Models & Constant Variables")
face_rec = ArcFace.ArcFace()

name_dict=list()
FULL_REGISTER_LIST=list()
FACE_COUNT_VERIFICATION = {}
TEMPORARY_SENT_RECORD = {}
DETECTION_SENSITIVITY = 0.8
MP_MODEL_SELECTION = 1 # 0 or 1
#CAMERA_ID = 0
CAMERA_ID = "rtsp://admin:abc12345@192.168.1.5:554/Stream/Channels/101"
VERIFICATION_COUNT = 3
VERIFICATION_PERIOD = 2 #min
ALLOW_SCAN_INTERVAL = 1
FACE_DISTANCE_THRESHOLD = 0.80

# Load Detection Model
mpFaceDetection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(DETECTION_SENSITIVITY, MP_MODEL_SELECTION)
logger.debug("Successfully Loaded Face Models & Constant Variables")

cap = WebcamVideoStream(src=CAMERA_ID).start()

#emb1 = face_rec.calc_emb('chua.jpeg')
#emb2 = face_rec.calc_emb('chua2.jpeg')
#emb3 = face_rec.calc_emb('jack.jpeg')

#path = 'C:/Users/user/Desktop/school-multiprocessing-facerec/face'
#path = "/Users/p2digital/Documents/school-multiprocessing-facerec/testing/face"
path = "/home/p2d/Documents/CCTV_FACE_PROJECT/school-face-recognition-python/face"

logger.debug("Loading Face Photo and Extracting Face Features")
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
    except Exception as e:
        logger.warning(type(e).__name__, __file__, e.__traceback__.tb_lineno)
        print(
            type(e).__name__,          # TypeError
            __file__,                  # /tmp/example.py
            e.__traceback__.tb_lineno  # 2
        )
logger.debug("Successfully Loaded Face Photo and Extracting Face Features")

emb_list = FULL_REGISTER_LIST

unique_name_dict = list(set(name_dict))
print(f"Raw Register: {name_dict}")
print(f"Unique Register: {unique_name_dict}")
logger.debug(f"Unique Register: {unique_name_dict}, Count: {len(unique_name_dict)}")

while True:
    try:
        frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #width, height = frame.shape[:2]
        height, width, channels = frame.shape
        #print(width, height, 1280, 720, 16:9)

        detections = faceDetection.process(rgb_frame)
        if detections.detections:
            for each_det in detections.detections:
                mp_drawing.draw_detection(frame, each_det)
                #print(detections.detections[0].location_data.relative_bounding_box)
                #r_bounding_box = detections.detections[0].location_data.relative_bounding_box
                r_bounding_box = each_det.location_data.relative_bounding_box
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
                min_value = np. amin(distance_list)

                if min_value < FACE_DISTANCE_THRESHOLD:
                    name_with_devuid = name_dict[idx]
                    #print(f"{name_dict[idx]}, {min_value}")
                    #if name_dict[idx] != 'Chua_30038':
                    #    print(distance_list)
                    
                    name = name_with_devuid.split('_')[0]
                    devuid = int(name_with_devuid.split('_')[1])
                    print(name, devuid, min_value)

                    if name in FACE_COUNT_VERIFICATION:
                        reset_timer_dif = datetime.datetime.now() - FACE_COUNT_VERIFICATION[name]['time']
                        reset_timer_sec = reset_timer_dif.total_seconds()
                        reset_timer_min = reset_timer_sec/60

                        if reset_timer_min >= VERIFICATION_PERIOD:
                            FACE_COUNT_VERIFICATION[name] = {}
                            FACE_COUNT_VERIFICATION[name]['count'] = 0
                            FACE_COUNT_VERIFICATION[name]['time'] = datetime.datetime.now()
                            print(f"{name} had passed {VERIFICATION_PERIOD}min, resetting its count...")
                        else:
                            FACE_COUNT_VERIFICATION[name]['count'] = reset_count(FACE_COUNT_VERIFICATION[name]['count'], VERIFICATION_COUNT)
                            FACE_COUNT_VERIFICATION[name]['count'] += 1
                    else:
                        FACE_COUNT_VERIFICATION[name] = {}
                        FACE_COUNT_VERIFICATION[name]['count'] = 1
                        FACE_COUNT_VERIFICATION[name]['time'] = datetime.datetime.now()
                        print(f"Created count verififcation for {name}")

                    if (name in FACE_COUNT_VERIFICATION) and (FACE_COUNT_VERIFICATION[name]['count'] > VERIFICATION_COUNT):
                        if name in TEMPORARY_SENT_RECORD:
                            time_diff = datetime.datetime.now() - TEMPORARY_SENT_RECORD[name]
                            time_sec = time_diff.total_seconds()
                            time_min = time_sec/60
                            if time_min <= ALLOW_SCAN_INTERVAL:
                                pass
                            else:
                                print(f"Attendance Success for {name}, {devuid}")
                                logger.info(f"Successfully push glog for Name: {name}, Enrollid: {devuid}, Time: {datetime.datetime.now()}")
                                TEMPORARY_SENT_RECORD[name] = datetime.datetime.now()
                        else:
                            print(f"Attendance Success for {name}, {devuid}")
                            logger.info(f"Successfully push glog for Name: {name}, Enrollid: {devuid}, Time: {datetime.datetime.now()}")
                            TEMPORARY_SENT_RECORD[name] = datetime.datetime.now()
                    else:
                        pass

                else:
                    pass
    
        cv2.imshow("webcam", frame)
    except Exception as e:
        logger.error(type(e).__name__, __file__, e.__traceback__.tb_lineno)
        print(
            type(e).__name__,          # TypeError
            __file__,                  # /tmp/example.py
            e.__traceback__.tb_lineno  # 2
        )
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow('Image')