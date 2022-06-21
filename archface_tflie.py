import cv2
import time

'''
from arcface import ArcFace
face_rec = ArcFace.ArcFace()
# time tracking


#emb1 = face_rec.calc_emb("jack.jpeg")


start_time = time.time()
emb2 = face_rec.calc_emb("john.jpeg")
emb3 = face_rec.calc_emb("jenny.jpeg")
distance = face_rec.get_distance_embeddings(emb2, emb3)
print(distance)
end_time = time.time() - start_time
print(f"Time Used: {end_time} second")

del(emb2)
del(face_rec)
del(emb3)
del(distance)
'''
from keras_facenet import FaceNet
embedder = FaceNet()

start_time = time.perf_counter()


img = cv2.imread('jack.jpeg')
#img2 = cv2.imread('john.jpeg')
#img3 = cv2.imread('jenny.jpeg')
emb4 = embedder.extract(img, threshold=0.95)[0]['embedding']
#emb5 = embedder.extract(img2, threshold=0.95)[0]['embedding']
#emb6 = embedder.extract(img2, threshold=0.95)[0]['embedding']

end_time = time.perf_counter()
print(end_time - start_time, "seconds")


'''
from keras_facenet import FaceNet
embedder = FaceNet()
img = cv2.imread('jack.jpeg')
emb4 = embedder.extract(img, threshold=0.95)[0]['embedding']
print(emb4)
'''


