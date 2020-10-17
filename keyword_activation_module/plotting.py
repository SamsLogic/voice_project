import wave
import numpy as np
import matplotlib.pyplot as plt

DPI = 200

plt.rcParams.update({'font.size': 5})
counter = 1
for l in range(15):
    fig = plt.figure(figsize=(10,10))
    spec = fig.add_gridspec(ncols=10,nrows=10,width_ratios=[2,2,2,2,2,2,2,2,2,2],height_ratios=[2,2,2,2,2,2,2,2,2,2])
    
    
    for j in range(10):
        for k in range(10):
            a = wave.open(f'recordings/{counter}.wav','r')
            b = []
            for i in range(184):
                au = a.readframes(256)
                au = np.fromstring(au,np.int16)
                b.append(au)
            b = np.array(b,np.int16)
            axs = fig.add_subplot(spec[j,k])
            axs.set_title(f'{counter}.wav')
            axs.imshow(b,cmap='gray')
            counter+=1
    
    fig.savefig(f"saved_images{l+1}.jpg", dpi=DPI)
