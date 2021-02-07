##scp brain.py pi@192.168.100.6:Project_V/brain/brain.py
##scp hearing_in_back_01.py pi@192.168.100.6:Project_V/brain/hearing.py
from datetime import date
import numpy as np
import pyaudio
import wave
from sklearn.preprocessing import MinMaxScaler

SAMPLE_RATE = 16000
CHANNELS = 4
RECORD_SECONDS = 1
BUFFER_SIZE = 8000
SAMPLE_WIDTH = 2
DIM = (SAMPLE_RATE,1)
TODAY = date.today()

scaler = MinMaxScaler(feature_range=(-1,1))
a = np.zeros((16000,1),dtype = np.int16)
a[0] = 32767
a[1] = -32767
scaler.fit(a)

def hear(q):
    p = pyaudio.PyAudio()
    stream = p.open(format= p.get_format_from_width(SAMPLE_WIDTH),
                channels=CHANNELS,
                rate= SAMPLE_RATE,
                input=True,
                output=True,
                input_device_index = 0,
                frames_per_buffer=BUFFER_SIZE)
    print('Hearing')
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
                stream.write(data)
            frames0 = np.array(frames0,dtype=np.int16)
            frames1 = np.array(frames1,dtype=np.int16)
            frames2 = np.array(frames2,dtype=np.int16)
            frames3 = np.array(frames3,dtype=np.int16)
            frames0 = np.reshape(frames0,DIM)
            frames1 = np.reshape(frames1,DIM)
            frames2 = np.reshape(frames2,DIM)
            frames3 = np.reshape(frames3,DIM)
            q.put(np.array([frames0,frames1,frames2,frames3],dtype=np.int16))
        except KeyboardInterrupt:
            stream.stop_stream()
            stream.close()
            p.terminate()

def listen(q,model):
    print('Listening')
    while True:
        voice = q.get()
        print(voice[:1].shape)
        keyword_detected = 0
        voice_1 = scaler.transform(voice[0])
        voice_1 = np.reshape(voice_1,(1,16000,1))
        pred = model.predict(voice_1)
        print(pred)
        if pred[0] > 0.5: #or pred[1]>0.9 or pred[2]>0.9 or pred[3]>0.9:
            break
    return 1, voice

