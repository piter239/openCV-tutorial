import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
start = int(time.time())
out = cv2.VideoWriter('./videos/outpy' + str(start) + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
def update_log_video():
    global start, out
    out.release()
    start = int(time.time())
    out = cv2.VideoWriter('./videos/outpy' + str(start) + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(True):
    ret, frame = cap.read()
    out.write(frame)
    cv2.imshow('frame', frame)
    t = 30 * 60
    if time.time() - start > t:
        update_log_video()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
