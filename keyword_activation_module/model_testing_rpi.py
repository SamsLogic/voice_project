import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

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

HOME_DIR = 'home/pi/Project_V/keyword_detection_module/'

p = pyaudio.PyAudio()

try:
    df = pd.read_csv('voice_data_testing.csv')
except:
    df = pd.DataFrame(columns=["name","label"])
    df.to_csv('voice_data_testing.csv',index=False)
    df = pd.read_csv('voice_data_testing.csv')
   

stream = p.open(format= p.get_format_from_width(SAMPLE_WIDTH),
                channels=CHANNELS,
                rate= SAMPLE_RATE,
                input=True,
                output=True,
                input_device_index = 0,
                frames_per_buffer=BUFFER_SIZE)

model = tf.keras.models.load_model('models/5th_version/voice_button_model_lstm.h5')
#model.load_weights('models/2nd version/voice_button_model_weights.h5py')

try:
    i = df.index.stop+1
except:
    i=1

print('Speak')

label = []
name = []

today = date.today()
try:
    os.system(f'mkdir test_recordings')
except:
    pass
try:
    os.system(f'mkdir test_recordings/{today}')
except:
    pass

while True:
    try:
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
        frames0 = np.array(frames0,dtype=np.int16)
        frames0 = np.reshape(frames0,(184,256))
        frames1 = np.array(frames1,dtype=np.int16)
        frames1 = np.reshape(frames1,(184,256))
        frames2 = np.array(frames2,dtype=np.int16)
        frames2 = np.reshape(frames2,(184,256))
        frames3 = np.array(frames3,dtype=np.int16)
        frames3 = np.reshape(frames3,(184,256))
        
        pred = model.predict(np.array([frames0,frames1,frames2,frames3],dtype=np.int16))
        
        print(pred)
        if pred[0] > 0.88:
            print("Hello")
            torf = int(input('Was it correct?(1: yes and 0: no)'))
            lb = 1
            if torf == 0:
                lb = 0
                
            wf = wave.open(f'test_recordings/{today}/{i}.wav','wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(p.get_format_from_width(SAMPLE_WIDTH)))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames0))
            wf.close()
    
            label.append(lb)
            name.append(f'{i}.wav')        
            i += 1
    
            wf = wave.open(f'test_recordings/{today}/{i}.wav','wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(p.get_format_from_width(SAMPLE_WIDTH)))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames1))
            wf.close()
    
            label.append(lb)
            name.append(f'{i}.wav')        
            i += 1
    
            wf = wave.open(f'test_recordings/{today}/{i}.wav','wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(p.get_format_from_width(SAMPLE_WIDTH)))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames2))
            wf.close()
    
            label.append(lb)
            name.append(f'{i}.wav')        
            i += 1
    
            wf = wave.open(f'test_recordings/{today}/{i}.wav','wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(p.get_format_from_width(SAMPLE_WIDTH)))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames3))
            wf.close()
    
            label.append(lb)
            name.append(f'{i}.wav')        
            i += 1
            
            df1 = {"name":name,"label":label}
            df1 = pd.DataFrame(df1)
            df1.to_csv('voice_data_testing.csv',mode='a',index=False,header=False)
            break
	
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
try:
    stream.stop_stream()
    stream.close()
    p.terminate()		
except:
    pass