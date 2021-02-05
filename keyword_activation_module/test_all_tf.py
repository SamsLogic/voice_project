
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

dir = 'D:\\voice_data_for_wake_word\\LibriSpeech\\consolidated_audio'
model_dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module'

BATCH_SIZE = 32
SAMPLE_RATE = 16000
CHANNELS = 1
EPOCHS = 1
RECORD_SECONDS = 1
#BUFFER_SIZE = 128
#FRAMES = fs//BUFFER_SIZE
DIM = (SAMPLE_RATE,1)
AUTOTUNE = tf.data.AUTOTUNE

test_files = tf.io.gfile.glob(dir + '\\test\\*\\*')

commands = ['0','1']

print('Test set size', len(test_files))

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return audio

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def get_audio_binary(file_path):
    audio_binary = tf.io.read_file(file_path)
    return audio_binary

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = get_audio_binary(file_path)
    waveform,_ = tf.audio.decode_wav(audio_binary)
    return waveform, tf.strings.to_number(label,out_type=tf.int32)

model = M.load_model(os.path.join(model_dir,'key_model_05022021.h5'))

test_ds = tf.data.Dataset.from_tensor_slices(test_files).map(get_waveform_and_label,num_parallel_calls=AUTOTUNE)

test_audio = []
test_label = []

for audio,label in test_ds:
    test_audio.append(audio.numpy())
    test_label.append(label.numpy())

test_audio = np.array(test_audio)
test_label = np.array(test_label)

print(test_audio.shape)
print(len(test_audio))

pred = model.predict(test_audio,batch_size=1,verbose=1)

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

plot_cm(test_label,pred)
