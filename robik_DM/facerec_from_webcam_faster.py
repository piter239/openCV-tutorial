import face_recognition
import cv2
import numpy as np
import time
#import pyttsx3
import glob
import os
import subprocess
import pickle

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)


video_capture = cv2.VideoCapture(0)



frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))



start = int(time.time())
out=cv2.VideoWriter('./videos/neuroOut'+str(start)+ '.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

def update_log_video():
    global start, out
    out.release()
    start = int(time.time())
    out=cv2.VideoWriter('./videos/outpy'+str(start)+ '.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))




#engine = pyttsx3.init()

# Load a sample picture and learn how to recognize it.
# Load a second sample picture and learn how to recognize it.

#simon_image = face_recognition.load_image_file("simon.jpg")
#simon_face_encoding = face_recognition.face_encodings(simon_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
#                        simon_face_encoding
]

known_face_names = [
#                    "Семён Литвинский"
]

def getName(str):
    ret = ""
    for j in str:
        if j == ".":
            break;
        if j == "/":
            ret = ""
        else:
            ret += j
    return ret

face_saves = dict()

# check if file dataset_faces.dat exists in current directory
# correctly handling linux/windows filenames
if not os.curdir+os.sep+'dataset_faces.dat' in glob.glob('./*'):
    print('we creates models. Wait please...')
    for i in glob.glob("kids/*"):
        str = i
        img = face_recognition.load_image_file(i)
        encods = face_recognition.face_encodings(img)
        if len(encods) == 0:
            print("Не вижу " + str)
        #print(i)
        else:
            known_face_encodings.append(face_recognition.face_encodings(img)[0])
            known_face_names.append(str)
            face_saves[i] = encods[0]
            print("Вижу " + str)
    with open('dataset_faces.dat', 'wb') as f:
        pickle.dump(face_saves, f)
else:
    print("LOADING faces from dataset_faces.dat file")
    with open('dataset_faces.dat', 'rb') as f:
        face_saves = pickle.load(f)
    for key in face_saves:
        val = face_saves[key]
        known_face_encodings.append(val)
        known_face_names.append(key)
last_hello = dict()

print("Number of known faces "+str(len(known_face_names)))
print(known_face_names)

for i in known_face_names:
    last_hello[i] = 0

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.circle(frame, ((left + right) // 2, (top + bottom) // 2), max(abs(left - right) // 2, abs(top - bottom) // 2) + 20, (255, 0, 0), 3)
        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom), (max(left + len(name) * 17, right), bottom + 35), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        t = 30 * 60
        t = 30
        #cv2.putText(frame, name, (left + 6, bottom - 6 + 35), font, 1.0, (255, 255, 255), 1)
        if name != "Unknown":
            if time.time() - last_hello[name] > t:
                #print("Hello " + name)
                #engine.say("Hello " + name)
                #engine.runAndWait()
                #os.system("echo Привет " + name + " |RHVoice-test -p Elena")
                #subprocess.Popen(["echo Привет " + getName(name) + " |RHVoice-test -p Elena"], shell=True)
                print(name)
                last_hello[name] = time.time()


    out.write(frame)
    t2 = 5 * 60
    if time.time() - start > t2:
        update_log_video()

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
out.release()
