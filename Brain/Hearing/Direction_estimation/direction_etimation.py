
import numpy as np
import math

SOUND_SPEED = 343.2
MIC_DISTANCE_4 = 0.08127
MAX_TDOA_4 = MIC_DISTANCE_4 / float(SOUND_SPEED)

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    
    sig = np.reshape(sig,(47104))
    refsig = np.reshape(refsig,(47104))
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    
    return tau, cc


def get_direction(buf,sample_rate=16000):
	best_guess = None
	
    MIC_GROUP_N = 2
    MIC_GROUP = [[0, 2], [1, 3]]
    
    tau = [0] * MIC_GROUP_N
    theta = [0] * MIC_GROUP_N
    for i, v in enumerate(MIC_GROUP):
        tau[i], _ = gcc_phat(buf[v[0]], buf[v[1]], fs=sample_rate, max_tau=MAX_TDOA_4, interp=1)
        theta[i] = math.asin(tau[i] / MAX_TDOA_4) * 180 / math.pi

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


    best_guess = (-best_guess + 120) % 360

	return best_guess