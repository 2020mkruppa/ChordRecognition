import numpy as np
import scipy
import librosa

#Refactored from C3S1_SpecLogFreq-Chromagram.ipynb

hopTime = 0.1
shortWindowTime = 0.2
longWindowTime = 0.8

chordDict = {'N':0, 
             'C':1, 'C#':2, 'D':3, 'D#':4, 'E':5, 'F':6, 'F#':7, 'G':8, 'G#':9, 'A':10, 'A#':11, 'B':12,
             'C:min':13, 'C#:min':14, 'D:min':15, 'D#:min':16, 'E:min':17, 'F:min':18, 'F#:min':19, 'G:min':20, 'G#:min':21, 'A:min':22, 'A#:min':23, 'B:min':24}
chordDictInv = {v : k for k, v in chordDict.items()}

enharmonics = {'Db':'C#', 'Eb':'D#', 'Gb':'F#', 'Ab':'G#', 'Bb':'A#'}
    

def getChordNumber(string):
    if string in chordDict:
        return chordDict[string]
    
    if string[:2] in enharmonics: #Test for enharmonics:
        string = enharmonics[string[:2]] + string[2:]
        if string in chordDict:
            return chordDict[string]
    
    #Remove 7th info and inversions, add, sus, etc.
    index = string.find(':')
    if index == -1:
        index = string.find('/')
    if index == -1:
        raise Exception("Could not find " + string) #return chordDict['N'] #This shouldn't happen ever, dummy return
    if string[:index] in chordDict:
        return chordDict[string[:index]]
    else:
        raise Exception("Could not find " + string) #return chordDict['N'] #This shouldn't happen ever either

#x is waveform from librosa.load()
def generateSpectrogram(x, Fs):
    H = int(hopTime * Fs)
    NShort = nearestPowerOfTwo(shortWindowTime * Fs)
    NLong = nearestPowerOfTwo(longWindowTime * Fs)
    return getData(x, Fs, NShort, H), getData(x, Fs, NLong, H)

###########################################################################################
###########################################################################################
###########################################################################################

def getData(x, Fs, N, H):
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N)
    Y, F_coef_pitch = compute_spec_log_freq(np.abs(X) ** 2, Fs, N) 
    T_coef = librosa.frames_to_time(np.arange(X.shape[1]), sr=Fs, hop_length=H)
    return [librosa.power_to_db(Y, ref=np.max), F_coef_pitch, T_coef]


def f_pitch(p, pitch_ref=69, freq_ref=440.0):
    return 2 ** ((p - pitch_ref) / 12) * freq_ref


def pool_pitch(p, Fs, N, pitch_ref=69, freq_ref=440.0):
    lower = f_pitch(p - 0.5, pitch_ref, freq_ref)
    upper = f_pitch(p + 0.5, pitch_ref, freq_ref)
    k = np.arange(N // 2 + 1)
    k_freq = k * Fs / N  # F_coef(k, Fs, N)
    mask = np.logical_and(lower <= k_freq, k_freq < upper)
    return k[mask]

def compute_spec_log_freq(Y, Fs, N):
    Y_LF = np.zeros((120, Y.shape[1]))
    for p in range(120):
        k = pool_pitch(p, Fs, N)
        Y_LF[p, :] = Y[k, :].sum(axis=0)
    F_coef_pitch = np.arange(120)
    return Y_LF, F_coef_pitch

def nearestPowerOfTwo(x):
    return int(np.exp2(np.ceil(np.log2(x))))