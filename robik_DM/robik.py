import face_recognition
import cv2
import numpy as np
import time
import glob
import os
# import subprocess
import pickle
# from pyimagesearch.centroidtracker import CentroidTracker
import random
import pyttsx3  # lets use some voice under Win10

# setting up text_to_speech engine (RHVoice preferred but not striktly)
tts_engine = pyttsx3.init()
tts_engine.setProperty('voice', 'ru')
voices = tts_engine.getProperty('voices')

t_props = ['voices', 'rate', 'volume']
t_eng = dict()
for p in t_props:
    t_eng[p] = tts_engine.getProperty(p)
print(t_eng)
tts_engine.setProperty('rate', 150)


# Попробовать установить предпочтительный голос
print("Number of availible voices: ", len(voices))
for voice in voices:
    print(voice.name)
    if voice.name == 'Aleksandr':
        print(voice.name + " set!")
        tts_engine.setProperty('voice', voice.id)


# Get a reference to webcam #0 (the default one)
frame_width = 640
frame_height = 480
start = int(time.time())
out = cv2.VideoWriter('./videos/neuroOut' + str(start) + '.avi',
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))


def update_log_video():
    global start, out
    out.release()
    start = int(time.time())
    out = cv2.VideoWriter('./videos/outpy' + str(start) + '.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))


# ct = CentroidTracker()
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)


# Create arrays of known face encodings and their names
known_face_encodings = []

known_face_names = []


def getName(strs):
    if strs[0] == ".":
        strs = strs[7:] # eliminating ./kids/
    ret = ""
    for j in strs:
        if j == ".":
            break
        if j == "/":
            ret = ""
        else:
            ret += j
    return ret


face_saves = dict()


def loadExistingDataset(face_saves):
# NB: object reference "face_saves" is passed by value
# so we have to add loaded dataset to it
# It is done for each key/value pair.
# Presently I don't know how to do it at confidence
# First idea TODO: return the loaded dictionary
    print('we load dataset of known people. 1sec...')
    with open('dataset_faces.dat', 'rb') as f:
        ex_face_saves = pickle.load(f)
        f.close()
    for key in ex_face_saves:
        val = ex_face_saves[key]
        face_saves[key] = val
        known_face_encodings.append(val)
        known_face_names.append(key)


loadNewModels = False   # load models from ./kids/ directory
addToExistingDataset = True
if loadNewModels:
    if addToExistingDataset:
        loadExistingDataset(face_saves)
    print("existing dataset loaded. # of known faces ", len(face_saves))
    print('we load new models. Wait please...')
    for i in glob.glob("./kids/*"):
        str = getName(i)
        img = face_recognition.load_image_file(i)
        encods = face_recognition.face_encodings(img)
        if len(encods) == 0:
            print("Не вижу " + str)
        # print(i)
        else:
            known_face_encodings.append(
                face_recognition.face_encodings(img)[0])
            known_face_names.append(str)
            face_saves[i] = encods[0]
            print("Вижу " + str)
        print("existing dataset loaded.1 # of known faces ", len(face_saves))
        print("existing dataset loaded.2 # of known faces ", len(face_saves))
    with open('dataset_faces.dat', 'wb') as f:
        pickle.dump(face_saves, f)
        f.close
else:
    loadExistingDataset(face_saves)
    print("existing dataset loaded.3 # of known faces ", len(face_saves))

# remember a list of greetings
greetings = [
    " Привет! ",
    " Меня зовут рОбик ",
    " Здравствуй, друг! ",
    " Приветствую! ",
    " Рад встрече, меня зовут рОбик ",
    " Привет, давай дружить! ",
    " Привет, Я робот рОбик "
]

NY = [
    " С наступающим ПРАЗДНИКОМ!",
    " С Новым годом!",
    " Желаю тебе всего наилучшего!",
    " Пусть сбудутся твои самые заветные мечты",
    " Пусть тебе мечтается, и пусть мечты - сбываются",
    " Желаю тебе здоровья и счастья!",
    " Желаю тебе  узнать много нового и интересного",
    " Пусть у тебя станет ещё больше настоящих друзей"
]


# remember a list of personal greetings
personal_greetings = [
    "Я думаю, ты {}. Подскажи, угадал ли я.",
    # "Я думаю, ты {}. Намекни, угадал ли я.",
    "Я думаю, ты {}. Дай знать, так ли это",
    "Ты напоминаешь мне моего друга по имени {}",
    "Мне кажется, я тебя узнал! Ты {}. Кивни, если это так",
    "Я думаю, ты {}. Подскажи, угадал ли я.",
    "Твоё имя {}? Дай знать, если да!",
    "Мне кажется, тебя зовут {}. Кивни, если да!"
]


# minimal times between greetings
last_hello = dict()

last_hello["all"] = 0   # global
last_hello["Unknown"] = 0   # Unknown
for i in known_face_names:
    last_hello[i] = 0   #

# set minimal time between consequent "Hello!" for Unknown
t_all = 10

# set minimal time between consequent "I know you!" for the same name
t = 30


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# for future refactoring
def DM_greet(str):
    ret


def utter_str(str='Кто и шутя и скоро возжелает пи узнать число тот знает?'):

    tts_engine.say(str, "first")
    # tts_engine.runAndWait() # this has to be called to read aloud. Blocks the thread!

from threading import Thread

def proceed_speech():
    tts_count = 0
    while True:
        tts_engine.runAndWait()
        tts_count += 1
        #print(tts_count)

#thread.start_new_thread(proceed_speech, ())
thread_tts = Thread(target=proceed_speech, args=())
thread_tts.start()

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        face_dists = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.45 and matches[best_match_index]:
                name = known_face_names[best_match_index]
                # print(name)
            face_dists.append(face_distances[best_match_index])
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    rects = []

# ПО УМУ ТУТ ПРОСТО НАДО ОТСОРТИРОВАТЬ ПО ДИСТАНЦИЯМ. Увы, моего знания Питона пока не хватает (Пётр)
    found_known_face = False
    for (top, right, bottom, left), name, dist in zip(face_locations, face_names, face_dists):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size

        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        rects.append(np.array([right, top, left, bottom]))

        # print(time.time() - last_hello["all"])

        # Draw a box around the face
        cv2.circle(frame, ((left + right) // 2, (top + bottom) // 2),
                   max(abs(left - right) // 2, abs(top - bottom) // 2) + 20, (255, 0, 0), 3)

        if name != "Unknown" and (not found_known_face):
            found_known_face = True
            # +t # pause with greetings no Unknown
            last_hello["all"] = time.time()
            (topN, rightN, bottomN, leftN) = (top, right, bottom, left)
            # and time.time() - last_hello["all"] > t_all:
            if time.time() - last_hello[name] > t:
                greeting = random.choice(
                    personal_greetings).format(getName(name))
                # random.choice(NY) + " . " +

                print(getName(name)+"\n")
                if not os.name == 'nt':
                    # subprocess.Popen(["echo  " + greeting + "  |RHVoice-test -p Elena"], shell=True)
                    utter_str(greeting)
                else:
                    utter_str(greeting)

                # last_hello["all"] = time.time() + t_all # pause with greetings no Unknown
                last_hello[name] = time.time()

                # remember POI
                found_known_face = True
                (topN, rightN, bottomN, leftN) = (top, right, bottom, left)

    # now loop through Unknowns. This is NOT a permanent solution
    for (top, right, bottom, left), name, dist in zip(face_locations, face_names, face_dists):

        if name == "Unknown" and not found_known_face:
            #print("detect " + name, " time_all ",
            #      (time.time() - last_hello["all"]))
            if time.time() - last_hello["all"] > t_all:
                greeting = random.choice(greetings) + " . " + random.choice(NY)

                if not os.name == 'nt':
                    # subprocess.Popen(["echo  " + greeting + "  |RHVoice-test -p Elena"], shell=True)
                    utter_str(greeting)
                else:
                    utter_str(greeting)
                last_hello["all"] = time.time()
                last_hello[name] = time.time()

    if found_known_face:
        cv2.circle(frame, ((leftN + rightN) // 2, (topN + bottomN) // 2),
                   max(abs(leftN - rightN) // 2, abs(topN - bottomN) // 2) + 20, (0, 255, 0), 4)

    out.write(frame)

    t2 = 5 * 60
    if time.time() - start > t2:
        update_log_video()

    cv2.imshow('Video', frame)
    #tts_engine.runAndWait()

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

out.release()
thread_tts.join()
