import time
from hearing.keyword_detection.hearing import hear, listen
from hearing.direction_estimation.direction_estimation import get_direction
from threading import Thread
from queue import Queue
import os

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

KEY_DIR = '/home/pi/Project_V/keyword_detection_module/'
KEYWORD_MODEL = M.load_model(os.path.join(KEY_DIR,'models/test_version/voice_button_model_lstm_2020-11-24_run.h5'))

if __name__ == '__main__':
    print('Starting')
    q = Queue()
    
    hearing_thread = Thread(target=hear,args=(q,),daemon=True)
    hearing_thread.start()
    time.sleep(1)
    keyword_detected,voice = listen(q,KEYWORD_MODEL)
    print(keyword_detected)
    print('Hello there')
    direction_voice = get_direction(voice)
    print(direction)
    
    