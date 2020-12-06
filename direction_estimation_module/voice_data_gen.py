

import pyaudio
import wave
import time
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from datetime import date

SAMPLE_RATE = 16000
CHANNELS = 4
RECORD_SECONDS = 3
BUFFER_SIZE = 256
SAMPLE_WIDTH = 2

p = pyaudio.PyAudio()

today = date.today()

try:
    df = pd.read_csv(f'voice_data_direction_{today}.csv')
except:
    df = pd.DataFrame(columns=["name","label","direction_0","direction_1","direction_2","direction_3","direction_4","direction_5","direction_6","direction_7"])
    df.to_csv(f'voice_data_direction_{today}.csv',index=False)
    df = pd.read_csv(f'voice_data_direction_{today}.csv')
    
stream = p.open(format= p.get_format_from_width(SAMPLE_WIDTH),
                channels=CHANNELS,
                rate= SAMPLE_RATE,
                input=True,
                output=True,
                input_device_index = 0,
                frames_per_buffer=BUFFER_SIZE)
                
label = []
name = []
direction_list = []
try:
    i = df.index.stop+1
except:
    i=1
print(i)

try:
    os.system('mkdir direction_recordings')
except:
    pass

try:
    os.system(f'mkdir direction_recordings/{today}')
except:
    pass

while True:
    try:
        direction = np.zeros((8),dtype=np.int16)
        stream.start_stream()
        frames0 = []
        frames1 = []
        frames2 = []
        frames3 = []
        print('SPEAK')
        
        for j in range(0, int(SAMPLE_RATE/BUFFER_SIZE*RECORD_SECONDS)):
            data = stream.read(BUFFER_SIZE,exception_on_overflow=False)
            a = np.fromstring(data,dtype=np.int16)[0::4]
            b = np.fromstring(data,dtype=np.int16)[1::4]
            c = np.fromstring(data,dtype=np.int16)[2::4]
            d = np.fromstring(data,dtype=np.int16)[3::4]
            frames0.append(a.tostring())
            frames1.append(b.tostring())
            frames2.append(c.tostring())
            frames3.append(d.tostring())
            stream.write(data)
        lb = int(input('Label: '))
        #lb = 1
        if lb == 0:
            continue
        direc = int(input('Direction(0,8): '))         
        #direc = 4
        direction[direc] = 1
        direction_list.append(direction)
        direction_list.append(direction)
        direction_list.append(direction)
        direction_list.append(direction)
        
        wf = wave.open(f'direction_recordings/{today}/{i}.wav','wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(SAMPLE_WIDTH)))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames0))
        wf.close()

        label.append(lb)
        name.append(f'{i}.wav')        
        
        i += 1

        wf = wave.open(f'direction_recordings/{today}/{i}.wav','wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(SAMPLE_WIDTH)))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames1))
        wf.close()

        label.append(lb)
        name.append(f'{i}.wav')
        
        i += 1

        wf = wave.open(f'direction_recordings/{today}/{i}.wav','wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(SAMPLE_WIDTH)))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames2))
        wf.close()

        label.append(lb)
        name.append(f'{i}.wav')
        
        i += 1

        wf = wave.open(f'direction_recordings/{today}/{i}.wav','wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(SAMPLE_WIDTH)))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames3))
        wf.close()

        label.append(lb)
        name.append(f'{i}.wav')
        
        i += 1
        
        print("Done")
        print(i)
        stream.stop_stream() 
        if i%250 == 0:
            print('250 done \n Stopping')
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        direction_list = np.array(direction_list,dtype=np.int16)
        df1 = {"name":name,"label":label,"direction_0":direction_list[:,0],"direction_1":direction_list[:,1],"direction_2":direction_list[:,2],"direction_3":direction_list[:,3],"direction_4":direction_list[:,4],"direction_5":direction_list[:,5],"direction_6":direction_list[:,6],"direction_7":direction_list[:,7]}
        df1 = pd.DataFrame(df1)
        df1.to_csv(f'voice_data_direction_{today}.csv',mode='a',index=False,header=False)
        print(df1)
        print('Exiting')
        break

stream.close()
p.terminate()











