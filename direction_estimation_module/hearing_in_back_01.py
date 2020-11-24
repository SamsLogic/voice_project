from threading import Thread
from queue import Queue

from datetime import date
import numpy as np
import pyaudio 
import wave

SAMPLE_RATE = 16000
CHANNELS = 4
RECORD_SECONDS = 3
BUFFER_SIZE = 47104
SAMPLE_WIDTH = 2
DIM = (184,256)
TODAY = date.today()

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
                data = stream.read(BUFFER_SIZE,exception_on_overflow=True)
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
            q.put([frames0,frames1,frames2,frames3])
        except KeyboardInterrupt:
            stream.stop_stream()
            stream.close()
            p.terminate()

def listen(model):
    while True:
        voice = q.get()
        pred = model.predict(voice)
        if pred == 1:
            break
    return 1

