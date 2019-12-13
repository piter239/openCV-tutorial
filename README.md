# some examples for openCV with Python - I am NOT the author, I just use this repository to collect useful sources

# installing python libraries:
#source: https://www.learnopencv.com/install-opencv-3-and-dlib-on-windows-python-only/

pip install numpy scipy matplotlib scikit-learn jupyter

pip install opencv-contrib-python

pip install cmake

pip install dlib



# Thing ready to show today

# detect faces with preset confidence
cd cd 1_deep-learning-face-detection
py detect_faces_video.py  -c 0.15


# Track faces with IDs in webcam or video files
cd simple-object-tracking
py object_tracker.py
