import dlib
from skimage import io
from scipy.spatial import distance
import argparse
from imutils.video import VideoStream
import imutils
import time

# Создаем модели для поиска и нахождения лиц в dlib
shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
sp = dlib.shape_predictor(shape_predictor_path)
face_recognition_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(face_recognition_model_path)
detector = dlib.get_frontal_face_detector()

# Загружаем первую фотографию
#img = io.imread('simon.jpg')

# # Находим лицо на фотографии
#dets = detector(img, 1)
#for k, d in enumerate(dets):
#    shape = sp(img, d)


# # Извлекаем дескриптор из лица
#face_descriptor1 = facerec.compute_face_descriptor(img, shape)
vs = VideoStream(src = 0).start()
win2 = dlib.image_window()

face_descriptor1 = None


while True:
    # # Загружаем и обрабатываем вторую фотографию
    img = vs.read()
    img = imutils.resize(img, height = 300)
    win2.clear_overlay()
    win2.set_image(img)
    dets_webcam = detector(img, 1)
    win2.clear_overlay()
    for k, d in enumerate(dets_webcam):
        shape = sp(img, d)
        face_descriptor2 = facerec.compute_face_descriptor(img, shape)
        if face_descriptor1 == None:
            face_descriptor1 = face_descriptor2
        a = distance.euclidean(face_descriptor1, face_descriptor2)
        #print(a)
        if a < 0.601:
            win2.add_overlay(d)
        time.sleep(0.1)
        break


#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
