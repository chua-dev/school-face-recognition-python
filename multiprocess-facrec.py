import cv2
import mediapipe as mp
import time
import multiprocessing
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras_facenet import FaceNet

cap = cv2.VideoCapture(1)
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.45,1)
embedder = FaceNet()

x_list = list()
y_list = list()
name_dict = dict()
path = 'D:/Users/Admin/Desktop/clientFaces/clients3'
i = 0
for folders in os.listdir(path):
    name_dict[i] = folders
    truePath = os.path.join(path,folders)
    for image in os.listdir(truePath):
        img = cv2.imread(os.path.join(truePath,image))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imgEncode = embedder.extract(img, threshold=0.95)[0]['embedding']
        x_list.append(imgEncode)
        y_list.append(i)
    i += 1

def face_detection(record,event):
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)
        if results.detections:
            for ids, detection in enumerate(results.detections):
                #mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                       int(bboxC.width * iw), int(bboxC.height * ih)
				
                record["bbox_data"] = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
				
                cv2.rectangle(img,bbox,(255,0,255),2)
            event.set()
        #cTime = time.time()
        #fps = 1/(cTime-pTime)
        #pTime = cTime
        #cv2.putText(img,f'fps {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
        #print(record["bbox_data"])
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('Image')

def recognize_face(encodeFace,x_list,y_list,name_dict,threshold):
    dist_list = list()
    for face in x_list:
        dist_list.append(np.linalg.norm(encodeFace-face))
    index1 = np.argmin(dist_list)
    #print(dist_list)
    #print(index1)
    if dist_list[index1] < threshold:
        index2 = y_list[index1]
        return name_dict[index2]
    else:
        pass

def face_recognition(record,event):
    while True:
        event.wait()
        img = record["bbox_data"]
        try:
            encode = embedder.embeddings([img])
            name = recognize_face(encode,x_list,y_list,name_dict,0.70)
            if name != None:
                print(name)
        except:
            pass
        event.clear()

if __name__ == "__main__":
    # creating processes
	manager = multiprocessing.Manager()
	events = multiprocessing.Event()

	records = manager.dict(bbox_data = '')
	
	p1 = multiprocessing.Process(target=face_detection, args=(records,events))
	p2 = multiprocessing.Process(target=face_recognition, args=(records,events))

	# starting process 1
	p1.start()
	# starting process 2
	p2.start()

	# wait until process 1 is finished
	p1.join()
	# wait until process 2 is finished
	p2.join()

	# both processes finished
	print("Done!")