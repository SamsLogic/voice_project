
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

# dir = 'D:\\voice_data_for_wake_word\\LibriSpeech\\consolidated_audio'
# csv_dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\csvs'
# model_dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module'
# train_data = pd.read_csv(os.path.join(csv_dir,'train.csv'))
# values = train_data.label.values

# BATCH_SIZE = 32
# SAMPLE_RATE = 16000
# CHANNELS = 1
# EPOCHS = 1
# RECORD_SECONDS = 1
# #BUFFER_SIZE = 128
# #FRAMES = fs//BUFFER_SIZE
# DIM = (SAMPLE_RATE,1)
# AUTOTUNE = tf.data.AUTOTUNE

# train_files = tf.io.gfile.glob(dir + '\\train\\*\\*')[:20000]
# dev_files = tf.io.gfile.glob(dir + '\\dev\\*\\*')
# test_files = tf.io.gfile.glob(dir + '\\test\\*\\*')

# commands = ['0','1']

# print(train_files[0])

# print('Training set size', len(train_files))
# print('Validation set size', len(dev_files))
# print('Test set size', len(test_files))

# def decode_audio(audio_binary):
    # audio, _ = tf.audio.decode_wav(audio_binary)
    # return audio

# def get_label(file_path):
    # parts = tf.strings.split(file_path, os.path.sep)
    # return parts[-2]

# def get_audio_binary(file_path):
    # audio_binary = tf.io.read_file(file_path)
    # return audio_binary

# def get_waveform_and_label(file_path):
    # label = get_label(file_path)
    # audio_binary = get_audio_binary(file_path)
    # waveform,_ = tf.audio.decode_wav(audio_binary)
    # return waveform, tf.strings.to_number(label,out_type=tf.int32)

# def get_spectrogram(waveform):
    # waveform = tf.cast(waveform, tf.float32)
    # spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    # spectrogram = tf.abs(spectrogram)
    # return spectrogram

# def get_spectrogram_and_label_id(audio, label):
    # spectrogram = get_spectrogram(audio)
    # spectrogram = tf.expand_dims(spectrogram, -1)
    # label_id = tf.argmax(label == commands)
    # return spectrogram, label_id
    
# def preprocess_dataset(files,count=None):
    # optimized_ds = tf.data.Dataset.from_tensor_slices(files).map(get_waveform_and_label,num_parallel_calls=AUTOTUNE).shuffle(1024).cache().repeat(count).batch(BATCH_SIZE,drop_remainder=True).prefetch(AUTOTUNE)
    # return optimized_ds
    

# test_ds = tf.data.Dataset.from_tensor_slices(test_files).map(get_waveform_and_label,num_parallel_calls=AUTOTUNE)

# METRICS = [
      # tf.keras.metrics.TruePositives(name='tp'),
      # tf.keras.metrics.FalsePositives(name='fp'),
      # tf.keras.metrics.TrueNegatives(name='tn'),
      # tf.keras.metrics.FalseNegatives(name='fn'), 
      # tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      # tf.keras.metrics.Precision(name='precision'),
      # tf.keras.metrics.Recall(name='recall'),
      # tf.keras.metrics.AUC(name='auc'),
# ]


# def build_model(metrics=METRICS,output_bias=None):
    # if output_bias != None:
        # output_bias = tf.keras.initializers.Constant(output_bias)
    
    # files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    # waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    # spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

    # for spectrogram, _ in spectrogram_ds.take(1):
        # input_shape = spectrogram.shape
    # print('Input shape:', input_shape)
    
    # num_labels = 1

    # #norm_layer = preprocessing.Normalization()
    # #norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))
    
    # model = models.Sequential([
        # L.Input(shape=input_shape),
        # preprocessing.Resizing(32, 32), #norm_layer,
        # L.Conv2D(32, 3, activation='relu'),
        # L.Conv2D(64, 3, activation='relu'),
        # L.MaxPooling2D(),
        # L.Dropout(0.25),
        # L.Flatten(),
        # L.Dense(128, activation='relu'),
        # L.Dropout(0.5),
        # L.Dense(num_labels,bias_initializer=output_bias,activation='sigmoid'),
    # ])
    
    # model.compile(
        # optimizer=tf.keras.optimizers.Adam(),
        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        # metrics=metrics)
    
    # return model

# def build_model_2(metrics=METRICS,output_bias=None):
    # if output_bias != None:
        # output_bias = tf.keras.initializers.Constant(output_bias)
    
    # files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    # waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    # for waveform, _ in waveform_ds.take(1):
        # input_shape = waveform.shape
    # print('Input shape:', input_shape)
    
    # inp = L.Input(shape=input_shape,name='Input')
    # X = L.Conv1D(8,13,strides=8,name='Conv1_1')(inp)
    # X = L.BatchNormalization()(X)
    # X = L.Activation('relu')(X)
    # X = L.MaxPooling1D(5,strides=2)(X)
    # X = L.Dropout(0.5)(X)
    # X = L.Conv1D(16,11,strides=4,name='Con2D_2')(X)
    # X = L.BatchNormalization()(X)
    # X = L.Activation('relu')(X)
    # X = L.MaxPooling1D(3,strides=2)(X)
    # X = L.Dropout(0.5)(X)
    # X = L.Conv1D(32,7,strides=2,name='Conv1_3')(X)
    # X = L.BatchNormalization()(X)
    # X = L.Activation('relu')(X)
    # X = L.MaxPooling1D(3,strides=2)(X)
    # X = L.Dropout(0.5)(X)
    # #X = L.LSTM(32,return_sequences=True)(X)
    # X = L.LSTM(32)(X)
    # out = L.Dense(1,activation='sigmoid',bias_initializer=output_bias,kernel_regularizer = tf.keras.regularizers.l2(1e-4))(X)
    
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    # initial_learning_rate=1e-3,
    # decay_steps=8000,
    # staircase=True,
    # decay_rate=0.9)
    
    # model = M.Model(inputs=[inp],outputs=[out])
    # bce = tf.keras.losses.BinaryCrossentropy()
    # adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
    # model.compile(loss=bce, optimizer='adam', metrics=metrics)
    
    # return model

# pos = len(tf.io.gfile.listdir(f'{dir}\\train\\{commands[1]}'))
# neg = len(tf.io.gfile.listdir(f'{dir}\\train\\{commands[0]}'))
# initial_bias = np.log([pos/neg])

# print(f'bias = {initial_bias}')

# model = build_model_2(output_bias = initial_bias)
# model.summary()

# log_dir = f"logs\\fit\\{datetime.datetime.now().strftime('%d%m%Y%M%S')}"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq = 1,update_freq='batch',profile_batch='500,520')

# class_weights = compute_class_weight('balanced',np.unique(values),train_data.label.values)
# train_data = []

def tensorflow_fit(mul):
    BATCH_SIZE = 32
    SAMPLE_RATE = 16000
    CHANNELS = 1
    EPOCHS = 20
    RECORD_SECONDS = 1
    #BUFFER_SIZE = 128
    #FRAMES = fs//BUFFER_SIZE
    DIM = (SAMPLE_RATE,1)
    AUTOTUNE = tf.data.AUTOTUNE
    
    dir = 'D:\\voice_data_for_wake_word\\LibriSpeech\\consolidated_audio'
    #csv_dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\csvs'
    model_dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module'
    
    train_files = tf.io.gfile.glob(dir + '\\train\\*\\*')
    train_files_1 = train_files[:10895]
    train_files_0 = train_files[10895*(mul+1):10895*(mul+3)]
    train_files = tf.concat([train_files_1,train_files_0],axis=0)
    dev_files = tf.io.gfile.glob(dir + '\\dev\\*\\*')
    
    log_dir = f"logs\\fit\\{datetime.datetime.now().strftime('%d%m%Y%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq = 1,update_freq='batch',profile_batch='500,520')
    
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

    def preprocess_dataset(files,count=None):
        optimized_ds = tf.data.Dataset.from_tensor_slices(files).map(get_waveform_and_label,num_parallel_calls=AUTOTUNE).cache().repeat(count).shuffle(2048).batch(BATCH_SIZE,drop_remainder=True).prefetch(AUTOTUNE)
        return optimized_ds
  
    model = M.load_model(os.path.join(model_dir,'key_model_again.h5'))
    print('Loaded Model: key_model_again.h5')
    train_ds = preprocess_dataset(train_files)
    dev_ds = preprocess_dataset(dev_files)
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        steps_per_epoch=len(train_files)//BATCH_SIZE,
                        validation_data = dev_ds,
                        validation_steps=len(dev_files)//BATCH_SIZE,
                        shuffle=True,
                        callbacks=[tensorboard_callback])

    print('Model Saving')
    model.save(f'key_model_again.h5') 
    print('Model Saved')
    #print('Model Saving Skipped')

    
if __name__ == "__main__":
    
    mulis = list(range(24))
    
    for muli in mulis:
        print('------------------------------------------------')
        print('------------------------------------------------')
        print(f'-------------Iteration {muli}---------------------')
        print('------------------------------------------------')
        print('------------------------------------------------')
        p = multiprocessing.Process(target=tensorflow_fit,args=[muli])
        p.start()
        p.join()
    
# model = M.load_model(f'key_model_2021-01-31.h5')
# model.summary()

# test_audio = []
# test_label = []

# # for audio,label in test_ds:
    # # test_audio.append(audio.numpy())
    # # test_label.append(label.numpy())

# # test_audio = np.array(test_audio)
# # test_label = np.array(test_label)

# # print(test_audio.shape)
# # print(len(test_audio))

# # pred = model.predict(test_audio,batch_size=1,verbose=1)

# def plot_cm(labels, predictions, p=0.5):
    # cm = tf.math.confusion_matrix(labels, predictions > p)
    # plt.figure(figsize=(5,5))
    # sns.heatmap(cm, annot=True, fmt="d")
    # plt.title('Confusion matrix @{:.2f}'.format(p))
    # plt.ylabel('Actual label')
    # plt.xlabel('Predicted label')
    # plt.show()
    # print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    # print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    # print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    # print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    # print('Total Fraudulent Transactions: ', np.sum(cm[1]))

# plot_cm(test_label,pred)
# print('waiting')
# time.sleep(10)

# print('Shutting down in 10s')
# time.sleep(10)
# os.system("shutdown /s /t 1") 
