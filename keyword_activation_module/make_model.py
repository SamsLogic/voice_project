
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.metrics as Metrics
from IPython import display
import wave
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import random
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K
from IPython import display
import multiprocessing

K.clear_session()
today = datetime.date.today()

#tf.debugging.set_log_device_placement(True)

physical_gpus = tf.config.experimental.list_physical_devices('GPU')
print(physical_gpus)

if physical_gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.optimizer.set_jit(True)
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

logical_gpus = tf.config.experimental.list_logical_devices('GPU')

DIM = (16000,1)
AUTOTUNE = tf.data.AUTOTUNE

commands = ['0','1']

dir = 'D:\\voice_data_for_wake_word\\LibriSpeech\\consolidated_audio'
csv_dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\csvs'
model_dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module'

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

def build_model_2(metrics=METRICS,output_bias=None):
    if output_bias != None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    
    inp = L.Input(shape=DIM,name='Input')
    X = L.Conv1D(16,19,strides=8,name='Conv1_1')(inp)
    X = L.BatchNormalization()(X)
    X = L.Activation('relu')(X)
    X = L.MaxPooling1D(5,strides=2)(X)
    X = L.Dropout(0.5)(X)
    X = L.Conv1D(32,17,strides=4,name='Con2D_2')(X)
    X = L.BatchNormalization()(X)
    X = L.Activation('relu')(X)
    X = L.MaxPooling1D(3,strides=2)(X)
    X = L.Dropout(0.5)(X)
    X = L.Conv1D(64,15,strides=2,name='Conv1_3')(X)
    X = L.BatchNormalization()(X)
    X = L.Activation('relu')(X)
    X = L.MaxPooling1D(3,strides=2)(X)
    X = L.Dropout(0.5)(X)
    X = L.LSTM(32,return_sequences=True)(X)
    X = L.LSTM(32)(X)
    out = L.Dense(1,activation='sigmoid',bias_initializer=output_bias,kernel_regularizer = tf.keras.regularizers.l2(1e-4))(X)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=8000,
    staircase=True,
    decay_rate=0.9)
    
    model = M.Model(inputs=[inp],outputs=[out])
    bce = tf.keras.losses.BinaryCrossentropy()
    adam = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(loss=bce, optimizer=adam, metrics=metrics)
    
    return model

pos = len(tf.io.gfile.listdir(f'{dir}\\train\\{commands[1]}'))
neg = len(tf.io.gfile.listdir(f'{dir}\\train\\{commands[0]}'))
initial_bias = np.log([pos/neg])

print(f'bias = {initial_bias}')

model = build_model_2(output_bias = initial_bias)
model.summary()
model.save(os.path.join(model_dir,'key_model_again.h5'))