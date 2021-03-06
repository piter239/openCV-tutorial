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

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file {}...",args["video"])
    vs = cv2.VideoCapture(args["video"])
# time.sleep(2.0)


# workaround for saving resulting video to file
# in the case of reading from file,wait after first frame
# this gives user the possibility to start recording
wait = (args["wait"] != 0)
# print("wait is ",wait)
#if not args.get("video", False):
#    wait = False
#    wait = True

img = io.imread('simon.jpg')
dets = detector(img, 1)
for k, d in enumerate(dets):
    shape = sp(img, d)
face_descriptor1 = facerec.compute_face_descriptor(img, shape)

auth = dict()
# loop over the frames from the video stream
while True:
    # read the next frame from the video stream and resize it
    frame = vs.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    frame = imutils.resize(frame, height=300)

    # if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # print(H, W)
    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, and initialize the list of
    # bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []
    isis = False
    # loop over the detections
    descriptors = dict()
    for i in range(0, detections.shape[2]):
        # filter out weak detections by ensuring the predicted
        # probability is greater than a minimum threshold
        if detections[0, 0, i, 2] > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))

            # draw a bounding box surrounding the object so we can
            # visualize it
            (startX, startY, endX, endY) = box.astype("int")
            startX -= dx
            endX += dx
            startY -= dy
            endY += dy
            #cv2.imshow("Frame", imutils.resize(frame[startX:endX, startY:endY], height = 300))
            img = frame[startX:endX, startY:endY]
            shape = None
            cenPos = ((startX + endX) // 2, (startY + endY) // 2)
            rad = max((endX - startX) // 2, (endY - startY) // 2)
            desc = None
            for d in detector(img, 1):
                shape = sp(img, d)
                desc = facerec.compute_face_descriptor(img, shape)
            if shape != None:
                #print(facerec.compute_face_descriptor(img, shape))
                isis = True
            if isis:
                cv2.circle(frame, cenPos, rad, (0, 255, 0), 2)
            else:
                cv2.circle(frame, cenPos, rad, (0, 0, 255), 2)
            text = "ID {}".format(i)
            if desc != None:
                a = distance.euclidean(face_descriptor1, desc)
            else:
                a = 1
            cv2.putText(frame, text, (cenPos[0] - 10, cenPos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if a < 0.601:
                cv2.circle(frame, cenPos, 4, (0, 255, 0), -1)
            else:
                cv2.circle(frame, cenPos, 4, (0, 0, 255), -1)

            #if isis:
            #    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            #else:
            #    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # show the output frame
    cv2.imshow("Frame", frame)
    if (wait):
        text = input("Start screen recording and hit Enter!")
        wait = False
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()
# otherwise, release the camera
else:
    vs.release()
cv2.destroyAllWindows()
