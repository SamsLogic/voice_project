
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.losses as Loss
import tensorflow.keras.optimizers as O

import wave
import pandas as pd
#import pyaudio
import soundfile as sf
import os
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
#import wave
import time
import datetime
import random
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K

K.clear_session()
today = datetime.date.today()

#tf.debugging.set_log_device_placement(True)

physical_gpus = tf.config.experimental.list_physical_devices('GPU')

if physical_gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            physical_gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

logical_gpus = tf.config.experimental.list_logical_devices('GPU')

dir = 'D:\\voice_data_for_wake_word\\LibriSpeech\\consolidated_audio\\recordings'
csv_dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\csvs'

audio, fs = sf.read(os.path.join(dir,'100_0.wav'))

BATCH_SIZE = 128
SAMPLE_RATE = fs
CHANNELS = 1
EPOCHS = 4
RECORD_SECONDS = 1
#BUFFER_SIZE = 128
#FRAMES = fs//BUFFER_SIZE
DIM = (fs,1)

train_data = pd.read_csv(os.path.join(csv_dir,'train.csv'))
test_data = pd.read_csv(os.path.join(csv_dir,'test.csv'))
dev_data = pd.read_csv(os.path.join(csv_dir,'dev.csv'))

# config = tf.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# tf.keras.backend.set_session(session)

class Data_Generator(tf.keras.utils.Sequence):
    
    def __init__(self,batch_size,music_ids,dir,dim,target=None,train=True,augment=False,shuffle=False):
        self.batch_size = batch_size
        self.music_ids = music_ids
        self.dir = dir
        self.augment = augment
        self.dim = dim
        self.target = target
        self.indices = range(len(self.music_ids))
        self.train = train
        self.shuffle = shuffle
    
    def on_epoch_end(self):
        return self.indices
    
    def getdata(self, music_id_list):
        X = np.zeros((self.batch_size,*self.dim))
        for i, m_id in enumerate(music_id_list):
                audio,_ = sf.read(os.path.join(self.dir,m_id))
                audio = np.reshape(audio,(self.dim[0],self.dim[1]))
                X[i,] = np.array(audio,dtype=np.int16)
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

def recall(y_true,y_pred):
    y_true = tf.cast(y_true,dtype=tf.float32)
    tot = tf.reduce_sum(y_true,axis=0)
    tp = tf.reduce_sum(y_true*y_pred,axis=0)
    fn = tf.reduce_sum(y_pred,axis=0)-tp
    return tp/(tp+fn)

def precision(y_true,y_pred):
    y_true = tf.cast(y_true,dtype=tf.float32)
    tot = tf.reduce_sum(y_true,axis=0)
    tp = tf.reduce_sum(tf.math.multiply(y_true,y_pred),axis=0)
    return tf.math.divide_no_nan(tp,tot)
 
def f1score(y_true,y_pred):
    y_true = tf.cast(y_true,dtype=tf.float32)
    pre = precision(y_true,y_pred)
    rec = recall(y_true,y_pred)
    return 2*tf.math.divide_no_nan(pre*rec,pre+rec)

def loss(y_true,y_pred):
    y_true = tf.cast(y_true,dtype=tf.float32)
    bce = Loss.BinaryCrossentropy()
    loss = precision(y_true,y_pred) - bce(y_true,y_pred) 
    return -loss

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


def build_model(metrics=METRICS,output_bias=None):
    if output_bias != None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    
    inp = L.Input(shape=DIM,name='Input')
    X = L.Conv1D(8,128,strides=8,name='Conv1_1')(inp)
    X = L.BatchNormalization()(X)
    X = L.Activation('relu')(X)
    X = L.MaxPooling1D(5,strides=2)(X)
    X = L.Dropout(0.5)(X)
    X = L.Conv1D(16,128,strides=4,name='Con2D_2')(X)
    X = L.BatchNormalization()(X)
    X = L.Activation('relu')(X)
    X = L.MaxPooling1D(3,strides=2)(X)
    X = L.Dropout(0.3)(X)
    X = L.Conv1D(32,64,strides=2,name='Conv1_3')(X)
    X = L.BatchNormalization()(X)
    X = L.Activation('relu')(X)
    X = L.MaxPooling1D(3,strides=2)(X)
    X = L.Dropout(0.3)(X)
    #X = L.LSTM(32,return_sequences=True)(X)
    X = L.LSTM(32)(X)
    out = L.Dense(1,activation='sigmoid',bias_initializer=output_bias)(X)
    
    model = M.Model(inputs=[inp],outputs=[out])
    bce = Loss.BinaryCrossentropy()
    #adam = O.Adam(learning_rate = 0.005)
    model.compile(loss=bce, optimizer='adam', metrics=metrics)
    
    return model

pos = train_data.label.value_counts()[1]
neg = train_data.label.value_counts()[0]
initial_bias = np.log([pos/neg])
print(f'bias = {initial_bias}')

# model = build_model(output_bias = initial_bias)
# model.summary()

train_gen = Data_Generator(BATCH_SIZE,train_data['name'],dir,DIM,target=train_data['label'],shuffle=True)
dev_gen = Data_Generator(BATCH_SIZE,dev_data['name'],dir,DIM,target=dev_data['label'],shuffle=True)
test_gen = Data_Generator(55,test_data['name'],dir,DIM,train=False)

log_dir = f"logs\\fit\\{datetime.datetime.now().strftime('%d%m%Y%M%S')}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,update_freq='batch',histogram_freq = 1)

class_weights = compute_class_weight('balanced',np.unique(train_data.label.values),train_data.label.values)

# history = model.fit(train_gen,
                    # steps_per_epoch = len(train_data)//BATCH_SIZE,
                    # epochs=EPOCHS,
                    # validation_data = dev_gen,
                    # shuffle=True,
                    # class_weight = {0:class_weights[0],1:class_weights[1]},
                    # callbacks=[tensorboard_callback])

# print('Model Saving')
# model.save(f'key_model_{today}.h5') 
# print('Model Saved')

model = M.load_model(f'key_model_2021-01-31.h5')
model.summary()

pred = model.predict(test_gen,verbose=1)

def plot_cm(labels, predictions, p=0.5):
    cm = tf.math.confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))

plot_cm(test_data.label.values,pred)

# print('Shutting down in 10s')
# time.sleep(10)
# os.system("shutdown /s /t 1") 
