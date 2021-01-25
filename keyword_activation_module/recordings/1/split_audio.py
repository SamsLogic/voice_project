import numpy as np
import wave
import os
import random
import pandas as pd

dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\1'
split_dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\1\\splited'

try:
    df = pd.read_csv(os.path.join(dir,'voice_data_l1.csv'))
except:
    df = pd.DataFrame(columns=["name","label"])
    df.to_csv(os.path.join(dir,'voice_data_l1.csv'),index=False)
    df = pd.read_csv(os.path.join(dir,'voice_data_l1.csv'))

label = np.array([])
name = []

a = os.listdir(dir)
a = a[:len(a)-3]
print("Started")

for i in a:
    #if random.randint(0,100) == 1:
    audio = wave.open(os.path.join(dir,i),'r')
    audio_f = np.fromstring(audio.readframes(audio.getnframes()),np.int16)
    audio_f = np.append(audio_f,np.zeros((896),np.int16))
    mean = []
    for j in range(5):
        #Retreiving Audio slice
        aud = audio_f[8000*j:8000*(j+2)]
        
        #Appending mean of positive signal values
        mean.append(np.mean(aud[aud>0]))
        
        #Saving Audio
        wf = wave.open(os.path.join(split_dir,f'{i[:len(i)-4]}_{j}.wav'),'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b''.join(aud))
        wf.close()
        
        #Add name to name list
        name.append(f'{i[:len(i)-4]}_{j}.wav')    
        
    max = np.max(mean)
    mean = np.array(mean)
    mean[mean<max/2] = 0        
    mean[mean>=max/2] = 1
    
    #Add label to label list
    label = np.append(label,mean)
    #else:
    #    continue
df1 = {"name":name,"label":label}
df1 = pd.DataFrame(df1)
df1.to_csv(os.path.join(dir,'voice_data_l1.csv'),mode='a',index=False,header=False)
print(df1)
print("Done")