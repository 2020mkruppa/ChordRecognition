import torch
import numpy as np
import os
import librosa
import Trainer as T
import FeatureExtractor as FE
import Preprocessor as P
from colorama import init
init(autoreset=True)
from colorama import Fore, Style
from scipy import stats

def netInference(x, Fs):              
    device = torch.device("cpu")
   
    model = T.Net().to(device)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    
    shortData, longData = FE.generateSpectrogram(x, Fs)
    
    processed = []
    for i in range(len(shortData[2])):
        processed.append([shortData[0][:,i], shortData[0][:,max(0, i - 1)], longData[0][:,i], longData[0][:,max(0, i - 1)]])    
    
    songFrames = np.array(processed, dtype=np.single) #Array of shape (numFrames, 4, 120)   
    songTensor = torch.from_numpy(np.reshape(songFrames, (-1, 4, 10, 12)))
    
    output = model(songTensor)
    output = output.argmax(dim=1).numpy()
    
    return output, medianBlur(output), shortData[2] #Argmax per frame which returns index number, key to chord dict
    
    
def medianBlur(seq):
    blurHalfWidth = 4
    blurred = []
    for i in range(len(seq)):
        value, count = stats.mode(seq[max(0, i - blurHalfWidth):min(len(seq), i + blurHalfWidth + 1)])
        blurred.append(value[0])
    return blurred
    
####################################################
####################################################

def findFiles(songTitle):
    strippedTitle = P.stripSongName(songTitle) 
    
    groundTruthFiles = dict() #Song name stripped -> path to annotations
    for subdir, dirs, files in os.walk('./The Beatles Annotations/chordlab/The Beatles'):
        for file in files:
            filename = str(file)
            if filename[-3:] != 'lab':
                continue
            title = P.stripSongName(filename[5:-3]) #Remove .wav
            groundTruthFiles[title] = os.path.join(subdir, file)


    for subdir, dirs, files in os.walk('./Beatles Songs'):
        for file in files:
            title = str(file)
            if title.find(' (Remastered 2009)') > -1:
                title = title[:-22] + '.wav'
            elif title.find(' (Medley _ Remastered 2009)') > -1:
                title = title[:-31] + '.wav'
            elif title[:14] == 'The Beatles - ':
                title = title[14:]
            elif title[:8] == 'Money (T':
                title = 'money.wav'

            if strippedTitle == P.stripSongName(title[:-4]):
                return os.path.join(subdir, file), groundTruthFiles[strippedTitle]  
    
    print('Could not find ' + strippedTitle)
    

def replaceMin(s):
    if s.find(':min') > -1:
        return s[:-4] + "m"
    return s
    
    
def drawLines(truth, inference, blurredInference):
    print("Truth:")
    print("Inference:")
    print("Blurred Inference:")
    print()
    sT = ""
    sI = ""
    sB = ""
    totalLength = 0
    for i in range(len(truth)):
        if totalLength > 180:
            print(sT)
            print(sI)
            print(sB)
            print()
            sT = ""
            sI = ""
            sB = ""
            totalLength = 0
        newTruth = replaceMin(FE.chordDictInv[truth[i]])
        newEst = replaceMin(FE.chordDictInv[inference[i]])
        newBlurred = replaceMin(FE.chordDictInv[blurredInference[i]])
        length = max(len(newTruth), max(len(newBlurred), len(newEst)))
        
        if FE.chordDictInv[truth[i]] != FE.chordDictInv[inference[i]]:
            sI += Style.BRIGHT + Fore.LIGHTRED_EX
        else:
            sI += Style.RESET_ALL
            
        if FE.chordDictInv[truth[i]] != FE.chordDictInv[blurredInference[i]]:
            sB += Style.BRIGHT + Fore.LIGHTRED_EX
        else:
            sB += Style.RESET_ALL
        sT += newTruth.ljust(length + 1)
        sI += newEst.ljust(length + 1)
        sB += newBlurred.ljust(length + 1)
        totalLength += length + 1
    print(sT)
    print(sI)
    print(sB)
    
    
def createConfusionMatrix(truth, inference):
    m = np.zeros((len(FE.chordDict), len(FE.chordDict)), dtype=int) #[truth, inf.]
    
    for i in range(len(truth)):
        m[truth[i], inference[i]] += 1
    
    space = 4
    label = "".ljust(space)
    for i in range(len(FE.chordDict)):
        label += replaceMin(FE.chordDictInv[i]).ljust(space)
    print("                                               Inference")
    print(label)
    
    for i in range(len(FE.chordDict)):
        s = replaceMin(FE.chordDictInv[i]).ljust(space)
        for j in range(len(FE.chordDict)):
            if i == j:
                s += Style.BRIGHT + Fore.BLACK
            elif m[i, j] > 0:
                s += Style.BRIGHT + Fore.LIGHTRED_EX
            else:
                s += Style.BRIGHT + Fore.WHITE
            s += str(m[i, j]).ljust(space)
        print(s)
    print("Truth")    
    
    F = []
    
    for i in range(len(FE.chordDict)): # i is class we are testing
        TP = m[i, i]
        FN = np.sum(m[i, :]) - m[i, i]
        FP = np.sum(m[:, i]) - m[i, i]
    
        if (TP + FN != 0) and (TP + FP != 0):
            R = TP / (TP + FN)
            P = TP / (TP + FP)
            F.append(2*P*R / (P + R))
    
    print()
    print("Average F: " + str(np.mean(np.array(F))))
    
def main():
    title = "Help!"
   
    musicFilePath, truthFilePath = findFiles(title)
    truth = [e[4] for e in P.processSong(musicFilePath, truthFilePath, np.zeros(len(FE.chordDict), dtype=int))]

    x, Fs = librosa.load(musicFilePath)
    inference, blurredInference, timeData = netInference(x, Fs)
  
    assert(len(truth) == len(inference))
    assert(len(truth) == len(blurredInference))
    drawLines(truth, inference, blurredInference)
    
    for i in range(3):
        print()
        
    createConfusionMatrix(truth, inference)    
    
    for i in range(3):
        print()
        
    createConfusionMatrix(truth, blurredInference)
    
    
        
if __name__ == '__main__':
    main()
    