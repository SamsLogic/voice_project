import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

import RPi.GPIO as GPIO
import time

from datetime import date
import numpy as np
import pyaudio 
import wave
import os

SAMPLE_RATE = 16000
CHANNELS = 4
RECORD_SECONDS = 3
BUFFER_SIZE = 47104
FRAMES = 47104
SAMPLE_WIDTH = 2
DIM = (184,256)

KEY_DIR = '/home/pi/Project_V/keyword_detection_module/'
DIREC_DIR = '/home/pi/Project_V/direction_detection_module/'
p = pyaudio.PyAudio()

GPIO.setmode(GPIO.BCM)

GPIO.setup(27,GPIO.OUT)
GPIO.setup(17,GPIO.OUT)

servo2 = GPIO.PWM(17,50)
servo1 = GPIO.PWM(27,50)
servo2.start(0)
servo1.start(0)

try:
    df = pd.read_csv(os.path.join(DIREC_DIR,'voice_data_dir_testing.csv'))
    key_df = pd.read_csv(os.path.join(KEY_DIR,'voice_data_testing.csv'))
except:
    df = pd.DataFrame(columns=["name","label","direction_0","direction_1","direction_2","direction_3","direction_4","direction_5","direction_6","direction_7"])
    df.to_csv(os.path.join(DIREC_DIR,'voice_data_dir_testing.csv'),index=False)
    df = pd.read_csv(os.path.join(DIREC_DIR,'voice_data_dir_testing.csv'))
    key_df = pd.DataFrame(columns=["name","label"])
    key_df.to_csv(os.path.join(KEY_DIR,'voice_data_testing.csv'),index=False)
    key_df = pd.read_csv(os.path.join(KEY_DIR,'voice_data_testing.csv')) 
    
    
   

stream = p.open(format= p.get_format_from_width(SAMPLE_WIDTH),
                channels=CHANNELS,
                rate= SAMPLE_RATE,
                input=True,
                output=True,
                input_device_index = 0,
                frames_per_buffer=BUFFER_SIZE)

model = tf.keras.models.load_model(os.path.join(KEY_DIR,'models/6th_version/voice_detection_model_lstm_update_2020-11-16.h5'))
#model.load_weights('models/2nd version/voice_button_model_weights.h5py')
dir_model = tf.keras.models.load_model(os.path.join(DIREC_DIR,'models/direction_model_lstm_v3.h5'))
try:
    i = df.index.stop+1
except:
    i=1
try:
    m = key_df.index.stop+1
except:
    m=1

print('Speak')


today = date.today()
try:
    os.system(f'mkdir /home/pi/Project_V/direction_detection_module/test_recordings')
    os.system(f'mkdir /home/pi/Project_V/keyword_detection_module/test_recordings')
except:
    pass
try:
    os.system(f'mkdir /home/pi/Project_V/direction_detection_module/test_recordings/{today}')
    os.system(f'mkdir /home/pi/Project_V/keyword_detection_module/test_recordings/{today}')
except:
    pass

def write_wave_file(dir,sample_width,sample_rate,data,num):
    loc = os.path.join(dir,f'test_recordings/{today}/{num}.wav')
    wf = wave.open(loc,'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(sample_width)))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(data))
    wf.close()

while True:
    try:
        #servo1.ChangeDutyCycle(0)
        label = []
        name = []
        key_label = []
        key_name = []
        stream.start_stream()
        frames0 = []
        frames1 = []
        frames2 = []
        frames3 = []
        for j in range(0, int(SAMPLE_RATE/BUFFER_SIZE*RECORD_SECONDS)):
            data = stream.read(BUFFER_SIZE,exception_on_overflow=False)
            a = np.fromstring(data,dtype=np.int16)[0::4]
            b = np.fromstring(data,dtype=np.int16)[1::4]
            c = np.fromstring(data,dtype=np.int16)[2::4]
            d = np.fromstring(data,dtype=np.int16)[3::4]
            frames0.append(a)
            frames1.append(b)
            frames2.append(c)
            frames3.append(d)
            stream.write(data)
        frames0 = np.array(frames0,dtype=np.int16)
        frames1 = np.array(frames1,dtype=np.int16)
        frames2 = np.array(frames2,dtype=np.int16)
        frames3 = np.array(frames3,dtype=np.int16)
        frames0 = np.reshape(frames0,(184,256))
        frames1 = np.reshape(frames1,(184,256))
        frames2 = np.reshape(frames2,(184,256))
        frames3 = np.reshape(frames3,(184,256))
        pred = model.predict(np.array([frames0,frames1,frames2,frames3],dtype=np.int16))
        print(pred)
        
        if pred[0] > 0.9 or pred[1]>0.9 or pred[2]>0.9 or pred[3]>0.9:
            print("Hello")
            frames0_dir = np.reshape(frames0,(184,256,1))
            frames1_dir = np.reshape(frames1,(184,256,1))
            frames2_dir = np.reshape(frames2,(184,256,1))
            frames3_dir = np.reshape(frames3,(184,256,1))
            dir_frame = [frames0_dir,frames1_dir,frames2_dir,frames3_dir]
            dir_frame = np.array(dir_frame,dtype=np.int16)
            dir_frame = np.transpose(dir_frame, (1,0,2,3))
            dir_frame = np.reshape(dir_frame,(1,*dir_frame.shape))
            print(dir_frame.shape)
            
            pred_dir = dir_model.predict(dir_frame)
            print(pred_dir)
            print('direction: ',np.argmax(pred_dir,axis=1))
            
            #torf = int(input('Was it correct (yes : 1 and no : 0)? '))
            torf = 0
            lb = 1
            if torf == 0:
                lb = 0
            
            write_wave_file(KEY_DIR,SAMPLE_WIDTH,SAMPLE_RATE,frames0,m)
            if lb == 1:
                write_wave_file(DIREC_DIR,SAMPLE_WIDTH,SAMPLE_RATE,frames0,i)
                i += 1
                name.append(f'{i}.wav')
                label.append(lb)
            key_label.append(lb)
            key_name.append(f'{m}.wav')        
            m += 1
    
            write_wave_file(KEY_DIR,SAMPLE_WIDTH,SAMPLE_RATE,frames1,m)
            if lb == 1:
                write_wave_file(DIREC_DIR,SAMPLE_WIDTH,SAMPLE_RATE,frames1,i)
                i += 1
                name.append(f'{i}.wav')
                label.append(lb)                
            key_label.append(lb)
            key_name.append(f'{m}.wav')        
            m += 1
            
            write_wave_file(KEY_DIR,SAMPLE_WIDTH,SAMPLE_RATE,frames2,m)
            if lb == 1:
                write_wave_file(DIREC_DIR,SAMPLE_WIDTH,SAMPLE_RATE,frames2,i)
                i += 1
                name.append(f'{i}.wav')
                label.append(lb)                
            key_label.append(lb)
            key_name.append(f'{m}.wav')        
            m += 1
    
            write_wave_file(KEY_DIR,SAMPLE_WIDTH,SAMPLE_RATE,frames3,m)
            if lb == 1:
                write_wave_file(DIREC_DIR,SAMPLE_WIDTH,SAMPLE_RATE,frames3,i)
                i += 1
                name.append(f'{i}.wav')                
                label.append(lb)
            key_label.append(lb)
            key_name.append(f'{m}.wav')        
            m += 1
            
            df1 = {"name":key_name,"label":key_label}
            df1 = pd.DataFrame(df1)
            df1.to_csv(os.path.join(KEY_DIR,'voice_data_testing.csv'),mode='a',index=False,header=False)
            
            #model.fit(np.array([frames0,frames1,frames2,frames3],dtype=np.float32),[lb,lb,lb,lb],batch_size=1,epochs=1)
            #model.save(os.path.join(KEY_DIR,'models/test_version/voice_button_model_lstm.h5'))
            #model = tf.keras.models.load_model(os.path.join(KEY_DIR,'models/test_version/voice_button_model_lstm.h5'))
            if lb == 1:

                direction_list = []
                direction = np.zeros((8),dtype=np.int16)
                direc = np.argmax(pred_dir,axis=1)
                #direc = int(input('direction of voice: '))
                servo1_angle_1 = 45*((1==direc)+(3==direc)+(4 ==direc)+(6==direc))
                servo1_angle_2 = 135*((2==direc)+(5==direc)+(7 ==direc)+(0==direc))
                servo2_angle_1 = 135*(((0==direc)+(1==direc)))
                servo2_angle_2 = 180*(((2==direc)+(3==direc)))
                servo2_angle_3 = 0*(((4==direc)+(5==direc)))
                servo2_angle_4 = 45*(((6==direc)+(7==direc)))
                #servo1.ChangeDutyCycle(2+((servo1_angle_1+servo1_angle_2)/18))
                #servo2.ChangeDutyCycle(2+(servo2_angle_1+servo2_angle_2+servo2_angle_3+servo2_angle_4)/18)
                #time.sleep(0.2)
                #servo1.ChangeDutyCycle(0)
                #servo2.ChangeDutyCycle(0)
                direction[direc] = 1
                direction_list.append(direction)
                direction_list.append(direction)
                direction_list.append(direction)
                direction_list.append(direction)
                direction_list = np.array(direction_list,dtype=np.int16)
                df1 = {"name":name,"label":label,"direction_0":direction_list[:,0],"direction_1":direction_list[:,1],"direction_2":direction_list[:,2],"direction_3":direction_list[:,3],"direction_4":direction_list[:,4],"direction_5":direction_list[:,5],"direction_6":direction_list[:,6],"direction_7":direction_list[:,7]}
                df1 = pd.DataFrame(df1)
                df1.to_csv(os.path.join(DIREC_DIR,'voice_data_dir_testing.csv'),mode='a',index=False,header=False)
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
        servo1.stop()
        servo2.stop()
try:
    stream.stop_stream()
    stream.close()
    servo1.stop()
    servo2.stop()
    p.terminate()		
except:
    pass