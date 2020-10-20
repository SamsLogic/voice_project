import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.optimizers as O

import wave
import pandas as pd
import pyaudio
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import wave

BATCH_SIZE = 16
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 3
BUFFER_SIZE = 256
DIM = (int(47104//BUFFER_SIZE),BUFFER_SIZE)
print('\n',DIM,'\n')
minmax = MinMaxScaler([0,1])

df = pd.read_csv('voice_data.csv')
p = pyaudio.PyAudio()

config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

class Data_Generator(tf.keras.utils.Sequence):
    
    def __init__(self,batch_size,music_ids,dim,target=None,train=True,augment=False):
        self.batch_size = batch_size
        self.music_ids = music_ids
        self.augment = augment
        self.dim = dim
        self.target = target
        self.indices = range(len(self.music_ids))
        self.train = train
    
    def on_epoch_end(self):
        return self.indices
    
    def getdata(self, music_id_list):
        X = np.zeros((self.batch_size,*self.dim))
        for i, m_id in enumerate(music_id_list):
            try:
                audio = wave.open(f'recordings/{m_id}','r')
                frames = []
                for j in range(self.dim[0]):
                    au = audio.readframes(self.dim[1])
                    frames.append(float(np.fromstring(au,np.int16))/255)
                if audio.getnframes != self.dim[0]*self.dim[1]:
                    for k in range(audio.getnframes()//BUFFER_SIZE,self.dim[0]):
                        frames[k] = np.zeros(256,dtype=np.int16)
            except:
                continue
            X[i,] = frames
            audio.close()
        return X
    
    def __getitem__(self,index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        music_id_list = [self.music_ids.values[k] for k in indices]
        X = self.getdata(music_id_list)
        if self.train == True:
            y = [self.target.values[k] for k in indices]
            y = np.array(y).astype(np.float32)
            return X,y
        return X
    
    def __len__(self):
        return int(np.floor(len(self.indices)/self.batch_size))

model = M.load_model('voice_button_model.h5')
model.summary()

print('\n',len(df.label.values),'\n')
train_gen = Data_Generator(BATCH_SIZE,
                           df.name,DIM,
                           target=df.label,
                           train=True)

pred = model.predict(train_gen,verbose=1)
print(len(pred))
print(pred.shape)

df['pred'] = pred

df.to_csv('voice_data_pred.csv',index=False)

tf.compat.v1.keras.backend.clear_session()
