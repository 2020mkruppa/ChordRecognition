import os
import FeatureExtractor as FE
import librosa
import pickle
import numpy as np

def processSong(musicFilePath, truthFilePath, distribution):
    x, Fs = librosa.load(musicFilePath)
    shortData, longData = FE.generateSpectrogram(x, Fs) #[librosa.power_to_db(Y, ref=np.max), F_coef_pitch, T_coef]
    
    truth = [] #[start time, end time, chord number]
    with open(truthFilePath, "r") as truthFile:
        for line in truthFile:
            line = line.rstrip()
            parts = line.split(' ')
            truth.append([float(parts[0]), float(parts[1]), FE.getChordNumber(parts[2])])
    
    data = []
    for i in range(len(shortData[2])):
        t = shortData[2][i]
        
        while (not (truth[0][0] <= t <= truth[0][1])) and len(truth) > 1: #Make sure there is at least 1 truth left 
            truth.pop(0)

        assert(len(shortData[0][:,i]) == 120)
        data.append([shortData[0][:,i], shortData[0][:,max(0, i - 1)], longData[0][:,i], longData[0][:,max(0, i - 1)], truth[0][2]])
        distribution[truth[0][2]] += 1
    return data
    
        
def stripSongName(title):
    return (''.join(e for e in title if e.isalnum())).lower() #Remove special characters, to lowercase
        

def main():
    groundTruthFiles = dict() #Song name stripped -> path to annotations
    for subdir, dirs, files in os.walk('./The Beatles Annotations/chordlab/The Beatles'):
        for file in files:
            filename = str(file)
            if filename[-3:] != 'lab':
                continue
            title = stripSongName(filename[5:-3]) #Remove .wav
            groundTruthFiles[title] = os.path.join(subdir, file)

    

    trainData = []
    trainDist = np.zeros(len(FE.chordDict), dtype=int)
    validationData = []
    validationDist = np.zeros(len(FE.chordDict), dtype=int)
    testData = []
    testDist = np.zeros(len(FE.chordDict), dtype=int)

    songCategories = [[], [], []]

    songNumber = 0
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

            strippedTitle = stripSongName(title[:-4]) #Remove .wav

            if strippedTitle not in groundTruthFiles:
                print('Could not find ' + strippedTitle)
                continue

            

            if songNumber % 10 < 8:
                songData = processSong(os.path.join(subdir, file), groundTruthFiles[strippedTitle], trainDist)
                trainData.extend(songData)
                songCategories[0].append(title[:-4])
            elif songNumber % 10 == 8:
                songData = processSong(os.path.join(subdir, file), groundTruthFiles[strippedTitle], validationDist)
                validationData.extend(songData)
                songCategories[1].append(title[:-4])
            else:
                songData = processSong(os.path.join(subdir, file), groundTruthFiles[strippedTitle], testDist)
                testData.extend(songData)
                songCategories[2].append(title[:-4])

            if songNumber % 5 == 0:
                print(songNumber)
            songNumber += 1


    #print(str(len(trainData)) + ' training vectors') 
    with open('trainData.pickle', 'wb') as trainFile:
        pickle.dump(trainData, trainFile)

    #print(str(len(validationData)) + ' validation vectors') 
    with open('validationData.pickle', 'wb') as validationFile:
        pickle.dump(validationData, validationFile)   

    #print(str(len(testData)) + ' test vectors') 
    with open('testData.pickle', 'wb') as testFile:
        pickle.dump(testData, testFile)

    f = open("songCategories.txt", "w")
    f.write("Training Songs:\n")
    for s in songCategories[0]:
        f.write("\t" + s + "\n")
    f.write("\n")

    f.write("Validation Songs:\n")
    for s in songCategories[1]:
        f.write("\t" + s + "\n")
    f.write("\n")

    f.write("Test Songs:\n")
    for s in songCategories[2]:
        f.write("\t" + s + "\n")
    f.write("\n")
    f.close()
    
    
    ind = np.flip(np.argsort(trainDist))
    
    colSpace = 20
    tabSpace = 8
    f1 = open("chordDistributions.txt", "w")
    f1.write("Chord Distributions\n\n")
    f1.write("".ljust(tabSpace) + "Training".ljust(colSpace) + "Validation".ljust(colSpace) + "Test".ljust(colSpace) + "\n")
    f1.write("".ljust(tabSpace) + str(len(trainData)).ljust(colSpace) + str(len(validationData)).ljust(colSpace) + str(len(testData)).ljust(colSpace) + "\n\n")
    for j in range(len(FE.chordDict)):
        i = ind[j]
        f1.write(FE.chordDictInv[i].ljust(tabSpace) + (str(trainDist[i]).ljust(6) + " -> " + ("%.4f" % (trainDist[i]/np.sum(trainDist)))).ljust(colSpace) + 
                                                      (str(validationDist[i]).ljust(6) + " -> " + ("%.4f" % (validationDist[i]/np.sum(validationDist)))).ljust(colSpace) + 
                                                      (str(testDist[i]).ljust(6) + " -> " + ("%.4f" % (testDist[i]/np.sum(testDist)))).ljust(colSpace) + "\n")
    f1.close()

if __name__ == '__main__':
    main()