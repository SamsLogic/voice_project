import os
import soundfile as sf
import pandas as pd
import numpy as np
import random
import shutil
import sys

random.seed(24)

dir1 = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\splited_recordings'
dir2 = 'D:\\voice_data_for_wake_word\\LibriSpeech'
dir3 = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\augment\\reference_audio'

data1 = pd.read_csv(os.path.join(dir1,'voice_data_new.csv'))
data2 = pd.read_csv(os.path.join(dir2,'voice_data_l2.csv'))

a = random.sample(list(data2['name'].values),500)
b = random.sample(list(data1[['name','label']].query('label == 1')['name'].values),50)

# for i in range(len(b)):
    # shutil.copy(os.path.join(dir1,b[i]),os.path.join(dir3,'1',b[i]))

# for j in range(len(a)):
    # shutil.copy(os.path.join(dir2,'consolidated_audio\\split',a[j]),os.path.join(dir3,'0',a[j]))
    
# print('copied')
    
   
for i in range(len(a)):
    au1,_ = sf.read(os.path.join(dir2,'consolidated_audio\\split',a[i]))
    for j in range(len(b)):
        au2,_ = sf.read(os.path.join(dir1,b[j]))
        au2 = au1+au2
        sf.write(f'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\augment\\augmented\\{i}_{j}.wav', au2,16000)
    sys.stdout.write(f'\rDone {i+1}/{len(a)}')
    sys.stdout.flush()