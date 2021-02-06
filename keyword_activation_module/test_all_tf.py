
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
import soundfile as sf

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
csv_dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\csvs'

BATCH_SIZE = 32
SAMPLE_RATE = 16000
CHANNELS = 1
EPOCHS = 1
RECORD_SECONDS = 1
#BUFFER_SIZE = 128
#FRAMES = fs//BUFFER_SIZE
DIM = (SAMPLE_RATE,1)
AUTOTUNE = tf.data.AUTOTUNE

test_files = tf.io.gfile.glob(model_dir+'\\new_test_results\\2021-02-06\\*')
test_data_labels = pd.read_csv(model_dir+'\\new_test_results\\voice_data_new.csv')

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
    audio_binary = get_audio_binary(file_path)
    waveform,_ = tf.audio.decode_wav(audio_binary)
    return waveform

model = M.load_model(os.path.join(model_dir,'key_model_04022021.h5'))
print(test_files)
print(test_data_labels)

test_ds = tf.data.Dataset.from_tensor_slices(test_files).map(get_waveform_and_label,num_parallel_calls=AUTOTUNE).batch(1)
for i in test_ds:
    print(i)
    a,_ = sf.read(test_files[0])
    print(a)
    break
pred = model.predict(test_ds,verbose=1)

print(pred)
test_label = test_data_labels.label.values 
def plot_cm(labels, predictions, p=0.9):
    cm = tf.math.confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])

plot_cm(test_label,pred)
