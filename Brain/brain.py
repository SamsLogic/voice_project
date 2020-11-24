import time
from hearing import hear, listen
from threading import Thread
from queue import Queue
import os

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

KEY_DIR = '/home/pi/Project_V/keyword_detection_module/'
DIREC_DIR = '/home/pi/Project_V/direction_detection_module/'

if __name__ == '__main__':
    print('Starting')
    q = Queue()
    keyword_model = M.load_model(os.path.join(KEY_DIR,'models/test_version/voice_button_model_lstm_2020-11-24_run.h5'))
    hearing_thread = Thread(target=hear,args=(q,),daemon=True)
    hearing_thread.start()
    time.sleep(1)
    keyword_detected = listen(q,keyword_model)
    print(keyword_detected)
    print('Hello there')