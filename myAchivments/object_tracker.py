from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import dlib
from scipy.spatial import distance
from skimage import io

shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'#
sp = dlib.shape_predictor(shape_predictor_path)#
face_recognition_model_path = 'dlib_face_recognition_resnet_model_v1.dat'#
facerec = dlib.face_recognition_model_v1(face_recognition_model_path)#
detector = dlib.get_frontal_face_detector()#

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-p", "--prototxt", default="deploy.prototxt",
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", help="path to Caffe pre-trained model",
                default="res10_300x300_ssd_iter_140000.caffemodel")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-w", "--wait", type=int, default=0,
                help="wait after the first frame is shown")

args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

dx = 20
dy = 20

last = dict()

def getGlobalID(localeID, isis):
    if isis:
        return 'Simon Litvinskiy'
    else:
        return localeID

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
else:
    print("[INFO] opening video file {}...",args["video"])
    vs = cv2.VideoCapture(args["video"])
wait = (args["wait"] != 0)
img = io.imread('simon.jpg')
dets = detector(img, 1)
for k, d in enumerate(dets):
    shape = sp(img, d)
face_descriptor1 = facerec.compute_face_descriptor(img, shape)
auth = dict()
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break
    frame = imutils.resize(frame, height=300)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []
    isis = False
    isis2 = False
    descriptors = dict()
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))
            (startX, startY, endX, endY) = box.astype("int")
            startX -= dx
            endX += dx
            startY -= dy
            endY += dy
            img = frame[startX:endX, startY:endY]
            shape = None
            cenPos = ((startX + endX) // 2, (startY + endY) // 2)
            rad = max((endX - startX) // 2, (endY - startY) // 2)
            for d in detector(img, 1):
                shape = sp(img, d)
                descriptors[cenPos] = facerec.compute_face_descriptor(img, shape)
            if shape != None:
                isis = True
            if isis:
                cv2.circle(frame, cenPos, rad, (0, 255, 0), 2)
            else:
                cv2.circle(frame, cenPos, rad, (0, 0, 255), 2)
    objects = ct.update(rects)
    for (objectID, centroid) in objects.items():
        globalID = getGlobalID(objectID, objectID in auth)
        text = "{}".format(globalID)
        if not globalID in last:
            print('open session for ', text)
        else:
            last[globalID] = 1
        if (centroid[0], centroid[1]) in descriptors:
            desc = descriptors[(centroid[0], centroid[1])]
            a = distance.euclidean(face_descriptor1, desc)
        else:
            a = 1
        if a < 0.601:
            auth[objectID] = True
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if objectID in auth:
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)


    for objectID in last:
        if last[objectID] == 0:
            print('close session for ', "{}".format(objectID))
    last = dict()
    for (objectID, centroid) in objects.items():
        last[getGlobalID(objectID, objectID in auth)] = 0
        
    cv2.imshow("Frame", frame)
    if (wait):
        text = input("Start screen recording and hit Enter!")
        wait = False
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()
