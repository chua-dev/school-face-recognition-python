from textwrap import fill
from tkinter import *
from tkinter import ttk
from tokenize import Name
from PIL import ImageTk, Image
import cv2
import mediapipe as mp
import multiprocessing
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#from keras_facenet import FaceNet
import datetime
from logging.handlers import TimedRotatingFileHandler
from imutils.video import WebcamVideoStream
import logging
import logging.handlers as handlers
import pathlib
import time
import os
import requests
from arcface import ArcFace

# Loading Arc Face model
face_rec = ArcFace.ArcFace()

# Reinitialize Logging Dir and File
pathlib.Path('./log').mkdir(parents=True, exist_ok=True) 

# detection and recognition init
PUSH_URL = "https://p2d.tkeeper.net/devices/push_glogs"
CAMERA_ID = "rtsp://admin:abc12345@192.168.1.5:554/Stream/Channels/101"
CONFIDENCE_COUNT = 3 # Frequency
CONFIDENCE_COUNT_MAXIMUM_TIME = 1 # Minit
ALLOW_SCAN_INTERVAL = 1 # Minit
FULL_REGISTER_LIST = list()

#cap = cv2.VideoCapture("rtsp://admin:abc12345@192.168.1.5:554/Stream/Channels/101")
cap = WebcamVideoStream(src=CAMERA_ID).start()
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.55,1)
#embedder = FaceNet()

# Setting Up Counter
face_count_before_confirm_record = {}
tempSentRecord = {}

######## Setting Up Logger ########
'''
logger = logging.getLogger('Time Attendance')
logger.setLevel(logging.INFO)

## Logging Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logHandler = handlers.TimedRotatingFileHandler('./log/attendance.log', when='M', interval=1, backupCount=2)
logHandler.setLevel(logging.INFO)

## Here we set our logHandler's formatter
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
'''

logging.basicConfig(filename="./log/attendance.log",
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='a')
 
# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
 
# Test messages
logger.debug("Initiating the Attendance System")
#logger.info("Just an information")
#logger.warning("Its a Warning")
#logger.error("Did you try to divide by zero")
#logger.critical("Internet is down")

######## Done Setting Logger ########

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

# recognition dictionary
x_list = list()
photo_list = list()
y_list = list()
name_dict = dict()
#path = 'D:/Users/Admin/Desktop/clientFaces/clients3'
#path = "/Users/p2digital/Documents/theface"
path = "/home/p2d/Documents/CCTV_FACE_PROJECT/school-face-recognition-python/face"
#path = "/Users/p2digital/Documents/school-multiprocessing-facerec/face"
#path = 'C:/Users/user/Desktop/school-multiprocessing-facerec/face'

i = 0
for folders in os.listdir(path):
    try:
        name_dict[i] = folders
        truePath = os.path.join(path,folders)
        for image in os.listdir(truePath):
            img = cv2.imread(os.path.join(truePath,image))
            height, width, channels = img.shape
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            detections = faceDetection.process(img)
            r_bounding_box = detections.detections[0].location_data.relative_bounding_box
            
            x, y, h, w = round(r_bounding_box.xmin * width) + 0, round(r_bounding_box.ymin * height) + 0, round(r_bounding_box.height * height) + 0, round(r_bounding_box.width * width) + 0

            img = img[y:y+h, x:x+w]
            #name_dict.append(folders)
            #extracts = embedder.extract(img, threshold=0.95)
            #imgEncode = extracts[0]['embedding']
            each_emb = face_rec.calc_emb(img)
            FULL_REGISTER_LIST.append(each_emb)
            #cv2.imshow('image',img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            img = image_resize(img, height = 140)
            #x_list.append(imgEncode)
            photo_list.append(img)
            y_list.append(i)
        i += 1
    except:
        print('unbuildable folder')

print(FULL_REGISTER_LIST)
print(name_dict)


'''# for fullscreen
def get_display_size():
    init_root = Tk()
    init_root.update_idletasks()
    init_root.attributes('-fullscreen', True)
    init_root.state('iconic')
    height = init_root.winfo_screenheight()
    width = init_root.winfo_screenwidth()
    init_root.destroy()
    return height, width

HEIGHT, WIDTH = get_display_size()'''

# mainpage
root = Tk()
root.title('P2D Face Attendance System')
#root.iconbitmap("face.ico")
root.geometry(f"{800}x{600}")

##########################
# User Input Logic

#####################################

# video
wrapper_video = LabelFrame(root)
wrapper_video.place(relx=0.03,rely=0.05,relheight=0.9,relwidth=0.66)

label_video = Label(wrapper_video)
label_video.place(in_=wrapper_video,anchor='c',relx=0.5,rely=0.5)

# side bar
wrapper = LabelFrame(root)
wrapper.place(relx=0.72,rely=0.05,relheight=0.9,relwidth=0.25)

f_canvas = Canvas(wrapper)
#f_canvas.place(relx=0,rely=0,relwidth=1,relheight=1)

# relx calculation
#relx = 0.5-(photo_width/(2*wrapper.winfo_width()))
f_canvas.place(relx=0.26,rely=0,relwidth=1,relheight=1)

yscrollbar = ttk.Scrollbar(wrapper,orient='vertical',command=f_canvas.yview)
yscrollbar.pack(side="right",fill="y")

f_canvas.configure(yscrollcommand=yscrollbar.set)

f_canvas.bind('<Configure>',lambda e: f_canvas.configure(scrollregion = f_canvas.bbox('all')))

frame = Frame(f_canvas)
f_canvas.create_window((0,0),window=frame,anchor='nw')

'''
# img frame
for i in range(40):
    # img
    frame1 = LabelFrame(frame)
    image = Image.open("D:/Users/Admin/Desktop/Ahmad Izhan/Ahmad Izhan_12360_a.jpeg")
    photo = ImageTk.PhotoImage(image.resize((100, 100), Image.ANTIALIAS))
    label = Label(frame1,image=photo)
    label.image = photo
    label.pack(side='left')
    Label(frame1,text='Hello').pack(side='left')
    Button(frame1,text="Lol").pack(side='left')
    frame1.pack(padx=20,pady=20)
'''

# functions
'''
def recognize_face(encodeFace,x_list,y_list,name_dict,threshold):
    dist_list = list()
    for face in x_list:
        dist_list.append(np.linalg.norm(encodeFace-face))
    index1 = np.argmin(dist_list)
    #print(dist_list)
    #print(index1)
    if dist_list[index1] < threshold:
        index2 = y_list[index1]
        return name_dict[index2],index1
    else:
        pass
'''

def recognize_face_arcface(detect_face_emb, name_dict, threshold):
    all_emb_distance = list()
    for face_emb in FULL_REGISTER_LIST:
        distance = face_rec.get_distance_embeddings(detect_face_emb, face_emb)
        all_emb_distance.append(distance)
    
    print(all_emb_distance)
    lowest_index = np.argmin(all_emb_distance)
    lowest_value = all_emb_distance[lowest_index]
    if lowest_value < threshold:
        return name_dict[lowest_index], lowest_index, lowest_value
    else:
        pass


def face_detection(record,event):
    while True:
        #img = cap.read()[1]
        img = cap.read()
        img = image_resize(img, height = int(root.winfo_screenheight()/1.3))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = faceDetection.process(img)
        bbox_data = list()
        if results.detections:
            for ids, detection in enumerate(results.detections):
                #mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                        int(bboxC.width * iw), int(bboxC.height * ih)
                
                bbox_data.append(img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]])
                cv2.rectangle(img,bbox,(255,0,255),2)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        label_video['image'] = img
        record["bbox_data"] = bbox_data    
        event.set()
        if record["name_with_photo"] != None:
            frame1 = LabelFrame(frame)
            name, photo = record["name_with_photo"]

            # photo
            photo = ImageTk.PhotoImage(Image.fromarray(photo))
            photo_label = Label(frame1,image=photo)
            photo_label.image = photo
            photo_label.pack(side="top",pady=0)

            # name
            Label(frame1,text=name).pack(side='top',pady=5)

            # datetime
            now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            Label(frame1,text=now).pack(side="top",pady=0)
            
            #padx calculation
            #padx = int((wrapper.winfo_width()/2)-(photo_label.winfo_width()/2))
            
            frame1.pack(pady=20,side='top')
            #record["name"] = None
            record['name_with_photo'] = None

        f_canvas.configure(scrollregion = f_canvas.bbox('all'))
        #f_canvas.yview_moveto('1.0')
        root.update()

def face_recognition(record,event):
    while True:
        event.wait()
        img_list = record["bbox_data"]
        #logger.info("Test for info")
        try:
            for img in img_list:
                #encode = embedder.embeddings([img])
                emb = face_rec.calc_emb(img)
                #full_name, idx, distance = recognize_face(encode,x_list,y_list,name_dict,0.71)
                full_name, idx, distance = recognize_face_arcface(emb, name_dict, 0.90)
                print(full_name, distance)
                #if name != None:
                name = full_name.split('_')[0]
                name_devuid = int(full_name.split('_')[1])
                # Check if name already exist
                if name in face_count_before_confirm_record:
                    reset_timer_dif = datetime.datetime.now() - face_count_before_confirm_record[name]['time']
                    reset_timer_sec = reset_timer_dif.total_seconds()
                    reset_timer_min = reset_timer_sec/60
                    print(reset_timer_dif)
                    print(reset_timer_sec)
                    print(reset_timer_min)
                    if reset_timer_min >= CONFIDENCE_COUNT_MAXIMUM_TIME: #Count Reset Threshold
                        # Recreate Timer and Pass counting since > 2min
                        face_count_before_confirm_record[name] = {}
                        face_count_before_confirm_record[name]['count'] = 0
                        face_count_before_confirm_record[name]['time'] = datetime.datetime.now()
                        print(f"Reseting Count For {name} because over 2min")
                    else: # Else Proceed with the count
                        face_count_before_confirm_record[name]['count'] = reset_count(face_count_before_confirm_record[name]['count'])
                        face_count_before_confirm_record[name]['count'] += 1
                        #print(f"{name} has had count of {face_count_before_confirm_record[name]['count']}")
                        #print(face_count_before_confirm_record)

                # If name not exist, create
                else:
                    face_count_before_confirm_record[name] = {}
                    face_count_before_confirm_record[name]['count'] = 1
                    face_count_before_confirm_record[name]['time'] = datetime.datetime.now()
                    #print(f"Created face count dict for count {name}")
                    #print(face_count_before_confirm_record)

                # If name exist in face count and face count == 3, only allow to try scan in 
                if (name in face_count_before_confirm_record) and (face_count_before_confirm_record[name]['count'] > CONFIDENCE_COUNT):
                    # Check if datetime
                    if name in tempSentRecord:
                        time_diff = datetime.datetime.now() - tempSentRecord[name]
                        time_sec = time_diff.total_seconds()
                        time_min = time_sec/60
                        if time_min <= ALLOW_SCAN_INTERVAL: #minit
                            print(f'{name} still within the cooldown period')
                            pass
                        else:
                            #print(name)
                            #record["name"] = name
                            #record["photo"] = photo_list[idx]
                            #logger.info(f"Pushing glog for {name}...")
                            # POST REQUEST
                            record["name_with_photo"] = (name,photo_list[idx])
                            tempSentRecord[name] = datetime.datetime.now()
                            #f_canvas.configure(scrollregion = f_canvas.bbox('all'))
                            #f_canvas.yview_moveto('1.0')
                            now = datetime.datetime.now()
                            params = { 'apiKey': 123456, 'kod': 'tw5PGzDVL6ZHjbHj', 'enrollid': name_devuid, 'verifymode': 0, 'inoutmode': 0, 
                                'tahun': now.year, 
                                'bulan': now.month, 
                                'hari': now.day, 
                                'jam': now.hour, 
                                'minit': now.minute, 
                                'saat': now.second
                                }
                            response = requests.post(PUSH_URL, data = params)
                            if response.ok:
                                print(f"Scanned for {name}")
                                logger.info(f"Successfully push glog for Name: {name}, Enrollid: {name_devuid}, Time: {now}")
                            else:
                                print(f"Failed to push for {name}")
                                logger.error(f"Failed push glog for Name: {name}, Enrollid: {name_devuid}, Time: {now}")

                            
                            del(now)
                            del(params)
                    else:
                        record["name_with_photo"] = (name,photo_list[idx])
                        tempSentRecord[name] = datetime.datetime.now()
                        #f_canvas.configure(scrollregion = f_canvas.bbox('all'))
                        #f_canvas.yview_moveto('1.0')
                        now = datetime.datetime.now()
                        params = { 'apiKey': 123456, 'kod': 'tw5PGzDVL6ZHjbHj', 'enrollid': name_devuid, 'verifymode': 0, 'inoutmode': 0, 
                            'tahun': now.year, 
                            'bulan': now.month, 
                            'hari': now.day, 
                            'jam': now.hour, 
                            'minit': now.minute, 
                            'saat': now.second
                            }
                        response = requests.post(PUSH_URL, data = params)
                        if response.ok:
                            print(f"Scanned for {name}")
                            logger.info(f"Successfully push glog for Name: {name}, Enrollid: {name_devuid}, Time: {now}")
                        else:
                            print(f"Failed to push for {name}")
                            logger.error(f"Failed push glog for Name: {name}, Enrollid: {name_devuid}, Time: {now}")

                        
                        del(now)
                        del(params)
                else:
                    print("Doesnt valid for scanning before reaching count 3+1")

        except Exception as exception:
            #print("Exception: {}".format(type(exception).__name__))
            #print("Exception message: {}".format(exception))
            pass
        event.clear()

def reset_count(count):
    if count > CONFIDENCE_COUNT:
        return 0
    else:
        return count

# multiprocessing
if __name__ == "__main__":
    
    # creating processes
    manager = multiprocessing.Manager()
    events = multiprocessing.Event()

    records = manager.dict(bbox_data = '',name_with_photo = None)
	
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


#root.mainloop()