import os
from playsound import playsound
import numpy as np
import pandas as pd

dir = 'C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\1\\splited\\0'
data = pd.read_csv('C:\\Users\\Sambhav\\Desktop\\voice_\\rpi\\keyword_activation_module\\recordings\\1\\splited\\0\\voice_data_l1_copy.csv')
update_label = data.copy()
print("Start")
counter = 0
print(data)
for i in data.index:
    if data["updated"][i] != True:
        print(f"Playing audio {data['name'][i]}")
        playsound(os.path.join(dir,data['name'][i]))
        inp = int(input('Real label: '))
        update_label['label'][i] = inp
        update_label['updated'][i] = True 
        if counter%10 == 0:
            print("Saving...")
            df1 = pd.DataFrame(update_label)
            df1.to_csv(os.path.join(dir,'voice_data_l1_copy.csv'),index=False)
        counter += 1
    else:
        continue
print(df1)
print("Done")