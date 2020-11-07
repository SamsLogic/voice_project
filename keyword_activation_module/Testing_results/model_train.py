# -*- coding: utf-8 -*-
"""train_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Fp7ZySLSg7fv2BwmYhqvh-sEgU_e7kFQ
"""

#!apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
#!pip install pyaudio

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

#audio = wave.open('/content/drive/My Drive/voice_model_train/recordings/100.wav')
audio = wave.open('test_recordings/2.wav')
audio.getnframes()

BATCH_SIZE = 8
SAMPLE_RATE = 16000
CHANNELS = 1
EPOCHS = 30
RECORD_SECONDS = 3
BUFFER_SIZE = 256
FRAMES = audio.getnframes()
DIM = (int(FRAMES//BUFFER_SIZE),BUFFER_SIZE)

minmax = MinMaxScaler([0,1])

#df = pd.read_csv('/content/drive/My Drive/voice_model_train/voice_data.csv')
df = pd.read_csv('voice_data_testing.csv')

df.label.value_counts()

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
            #audio = wave.open(f'/content/drive/My Drive/voice_model_train/recordings/{m_id}','r')
            audio = wave.open(f'test_recordings/{m_id}','r')
            frames = []
            for j in range(self.dim[0]):
                au = audio.readframes(self.dim[1])
                au = np.fromstring(au,np.int16)
                frames.append(au)
            X[i,] = np.array(frames,dtype=np.int16)
            audio.close()
        return X
    
    def __getitem__(self,index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        music_id_list = [self.music_ids.values[k] for k in indices]
        X = self.getdata(music_id_list)
        #X = np.reshape(X,(self.batch_size,*self.dim,1))
        if self.train == True:
            y = [self.target.values[k] for k in indices]
            y = np.array(y).astype(np.int16)
            return X,y
        return X
    
    def __len__(self):
        return int(np.floor(len(self.indices)/self.batch_size))

model = M.load_model('../models/6th_version/voice_button_model_lstm_new.h5')

def lrfn(epoch, lr):
    lr_max = 0.1
    lr_min = 0.000001
    lr_decay = 0.85
    epoch_sust  = 2
    if epoch % epoch_sust == 0:
        lr = lr_max * lr_decay**((epoch+1)/0.19) 
        return lr
    else:
        return lr

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lrfn,verbose=1)

a = []
lr = 0.1
for i in range(EPOCHS):
  lr = lrfn(i,lr)
  a.append(lr)
plt.plot(a)
plt.show()

print('\n',len(df.label.values),'\n')

train_gen = Data_Generator(BATCH_SIZE,
                           df.name,DIM,
                           target=df.label,
                           train=True)

history = model.fit(train_gen,
                    epochs=EPOCHS,
                    steps_per_epoch=len(df.label.values)//BATCH_SIZE)

#model.save('/content/drive/My Drive/voice_model_train/voice_button_model_lstm.h5')
model.save('../models/6th_version/voice_button_model_lstm_update.h5')

tf.compat.v1.keras.backend.clear_session()

print(model.evaluate(train_gen,verbose=0))

