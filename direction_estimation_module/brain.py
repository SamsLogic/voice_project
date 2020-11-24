import time
from hearing_in_back_01.py import hear, listen
from threading import Thread
from queue import Queue
import os

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

KEY_DIR = '/home/pi/Project_V/keyword_detection_module/'
DIREC_DIR = '/home/pi/Project_V/direction_detection_module/'

if __init__ = '__main_':
    print('Starting')
    q = Queue()
    keyword_model = M.load_model(os.path.join(KEY_DIR,'models/6th_version/voice_button_model_lstm_2020-11-18_8.h5'))
    hearing_thread = Thread(target=hear,args=(q,))
    hearing_thread.start()
    keyword_detected = listen(q,keyword_model)
    print('Hello there')