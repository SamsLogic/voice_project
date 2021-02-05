import numpy as np
import wave
import os
import pandas as pd
import shutil

dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\1\\splited\\0'
data = pd.read_csv('C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\1\\splited\\0\\voice_data_l1_copy.csv')

print('Started')
counter=0

for i in data.index:
    if data["label"][i] == 0:
        shutil.move(os.path.join(dir,data["name"][i]),os.path.join(dir,f'0\\{data["name"][i]}'))
    else:
        shutil.move(os.path.join(dir,data["name"][i]),os.path.join(dir,f'1\\{data["name"][i]}'))
        counter+=1
print(f'actual count = {len(data["label"].values[data["label"].values == 1])}')
print(f'count = {counter}')
print('Done')