# # Пример верификации человека на фотографии с помощью библиотеки dlib
# **Верификация** - это задача определения,
# находится ли на изображении нужный нам человек, или нет.
# Мы будем решать задачу верификации человека на двух фотографиях.
# source adapted from
# https://github.com/sozykin/dlpython_course/blob/master/computer_vision/foto_comparison/foto_verification.ipynb

# usage:
# py foto_verification.py image1 image2


# Нам нужно будет определить, один ли еловек изображен на двух фотографиях.

# Предварительно обученные модели можно скачать по ссылкам:
# - http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# - http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
#
# Файлы с моделями нужно разархивировать и положить в каталог с этим скритом

import dlib
from skimage import io
from scipy.spatial import distance
import argparse

# Создаем модели для поиска и нахождения лиц в dlib

shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
sp = dlib.shape_predictor(shape_predictor_path)
face_recognition_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(face_recognition_model_path)
detector = dlib.get_frontal_face_detector()


# get filenames from command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("image1")
parser.add_argument("image2")
args = parser.parse_args()
print(args.image1, args.image2)


# Загружаем первую фотографию
img = io.imread(args.image1)

# Показываем фотографию средствами dlib
win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)

# # Находим лицо на фотографии
dets = detector(img, 1)

for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)


# # Извлекаем дескриптор из лица
face_descriptor1 = facerec.compute_face_descriptor(img, shape)

# Печатаем дексриптор
# print(face_descriptor1)


# # Загружаем и обрабатываем вторую фотографию
img = io.imread(args.image2)
win2 = dlib.image_window()
win2.clear_overlay()
win2.set_image(img)
dets_webcam = detector(img, 1)
for k, d in enumerate(dets_webcam):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win2.clear_overlay()
    win2.add_overlay(d)
    win2.add_overlay(shape)

face_descriptor2 = facerec.compute_face_descriptor(img, shape)


# # Рассчитываем Евклидово расстояние между двумя дексрипторами лиц
#
# В dlib рекомендуется использовать граничное значение Евклидова расстояния
# между дескрипторами лиц равное 0.6. Если Евклидово расстояние меньше 0.6,
# значит фотографии принадлежат одному человеку.
#
# С использованием такой метрики dlib обеспечивает точность 99.38% на тесте
# распознавания лиц Labeled Faces in the Wild. Подробности можно посмотреть
# по ссылке - http://dlib.net/face_recognition.py.html

a = distance.euclidean(face_descriptor1, face_descriptor2)
print(a)
input("Press any key to exit")
