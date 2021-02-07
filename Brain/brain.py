import time
from hearing.keyword_detection.hearing import hear, listen
from hearing.direction_estimation.direction_estimation import get_direction
from movement.head_movement.head_movement import head_servo_movement
from threading import Thread
from queue import Queue
import os

import tensorflow as tf

KEY_DIR = '/home/pi/Project_V/keyword_detection_module/'
KEYWORD_MODEL = tf.keras.models.load_model(os.path.join(KEY_DIR,'models/new_model/key_model_again.h5'))

if __name__ == '__main__':
    print('Starting')
    q = Queue()
    
    hearing_thread = Thread(target=hear,args=(q,),daemon=True)
    hearing_thread.start()
    time.sleep(1)
    #while True:
        #try:
    keyword_detected,voice = listen(q,KEYWORD_MODEL)
    print(f'Keyword detected : {keyword_detected}')
    direction_voice_x,direction_voice_y = get_direction(voice)
    print(direction_voice_x,direction_voice_y)
    #head_servo_movement(direction_voice_x,direction_voice_y)
    print('Hello there')
        #except:
            #break
    
    
