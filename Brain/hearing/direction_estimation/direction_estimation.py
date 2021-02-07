#import matplotlib.pyplot as plt
import numpy as np
import math
import wave

SOUND_SPEED = 343.2
MIC_DISTANCE_4 = 0.08127
MIC_DISTANCE_4_LAT = 0.05746

MAX_TDOA_4 = MIC_DISTANCE_4 / float(SOUND_SPEED)
MAX_TDOA_4_LAT = MIC_DISTANCE_4_LAT / float(SOUND_SPEED)

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    
    sig = np.reshape(sig,(16000))
    refsig = np.reshape(refsig,(16000))
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    #fig,ax = plt.subplots(5,figsize=(15,5))
    #ax[0].plot(np.linspace(0,len(SIG),len(SIG)),SIG)
    REFSIG = np.fft.rfft(refsig, n=n)
    #ax[1].plot(np.linspace(0,len(REFSIG),len(REFSIG)),REFSIG)
    R = SIG * np.conj(REFSIG)
    #ax[2].plot(np.linspace(0,len(R),len(R)),R)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    #ax[3].plot(np.linspace(0,len(cc),len(cc)),cc)
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    #ax[4].plot(np.linspace(0,len(cc),len(cc)),cc)
    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    #plt.show()
    return tau, cc


def get_direction(buf,sample_rate=16000):
    best_guess = None
    best_guess_lat = None
	
    MIC_GROUP_N = 2
    MIC_GROUP = [[0, 2], [1, 3]]
    
    tau = [0] * MIC_GROUP_N
    theta = [0] * MIC_GROUP_N
    tau_lat = [0] * MIC_GROUP_N
    theta_lat = [0] * MIC_GROUP_N
    
    for i, v in enumerate(MIC_GROUP):
        tau[i], _ = gcc_phat(buf[v[0]], buf[v[1]], fs=sample_rate, max_tau=MAX_TDOA_4, interp=1)
        theta[i] = math.asin(tau[i] / MAX_TDOA_4) * 180 / math.pi
        tau_lat[i], _ = gcc_phat(buf[v[0]], buf[v[1]], fs=sample_rate, max_tau=MAX_TDOA_4_LAT, interp=1)
        theta_lat[i] = math.asin(tau_lat[i] / MAX_TDOA_4_LAT) * 180 / math.pi

    if np.abs(theta[0]) < np.abs(theta[1]):
        if theta[1] > 0:
            best_guess = (theta[0] + 360) % 360
        else:
            best_guess = (180 - theta[0])
    else:
        if theta[0] < 0:
            best_guess = (theta[1] + 360) % 360
        else:
            best_guess = (180 - theta[1])

        best_guess = (best_guess + 90 + 180) % 360

    if np.abs(theta_lat[0]) < np.abs(theta_lat[1]):
        if theta_lat[1] > 0:
            best_guess_lat = (theta_lat[0] + 360) % 360
        else:
            best_guess_lat = (180 - theta_lat[0])
    else:
        if theta_lat[0] < 0:
            best_guess_lat = (theta_lat[1] + 360) % 360
        else:
            best_guess_lat = (180 - theta_lat[1])

        best_guess_lat = (best_guess_lat + 90 + 180) % 360

    best_guess = (-best_guess + 120) % 360
    best_guess_lat = (-best_guess_lat + 120) % 360
    best_guess_lat = np.abs((90*(best_guess_lat//90)) - best_guess_lat)
    return best_guess,best_guess_lat
    
def main():
    audio1 = wave.open('C:\\Users\Sambhav\Desktop\\rpi\direction_estimation_module\direction_recordings\\9.wav','r')
    audio2 = wave.open('C:\\Users\Sambhav\Desktop\\rpi\direction_estimation_module\direction_recordings\\10.wav','r')
    audio3 = wave.open('C:\\Users\Sambhav\Desktop\\rpi\direction_estimation_module\direction_recordings\\11.wav','r')
    audio4 = wave.open('C:\\Users\Sambhav\Desktop\\rpi\direction_estimation_module\direction_recordings\\12.wav','r')
    frames1 = []
    frames2 = []
    frames3 = []
    frames4 = []
    for j in range(128):
        au1 = audio1.readframes(125)
        au1 = np.fromstring(au1,np.int16)
        for k in au1:
            frames1.append(k)
        au2 = audio2.readframes(125)
        au2 = np.fromstring(au2,np.int16)
        for k in au2:
            frames2.append(k)
        au3 = audio3.readframes(125)
        au3 = np.fromstring(au3,np.int16)
        for k in au3:
            frames3.append(k)
        au4 = audio4.readframes(125)
        au4 = np.fromstring(au4,np.int16)
        for k in au4:
            frames4.append(k)
    direction,direction_z = get_direction([frames1,frames2,frames3, frames4])
    print(direction,direction_z)
    
if __name__ == '__main__':
    main()
