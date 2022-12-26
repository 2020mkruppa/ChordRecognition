import numpy as np
import os
import subprocess
import cv2
import librosa
import FeatureExtractor as FE
import Inference as I
import datetime, threading, time
from playsound import playsound
from PIL import ImageFont, ImageDraw, Image

def printInference(sequence):
    time.sleep(0.25)   
    base = time.time()
    while True:
        print(I.replaceMin(FE.chordDictInv[sequence[0][0]])) 
        del sequence[0]
        if len(sequence) == 0:
            break
        time.sleep((base + sequence[0][1]) - time.time())      
    
def getChordStrings(seq, i):
    s = []
    for j in range(i - 2, i + 3):
        if j < 0 or j >= len(seq):
            s.append("")
        else:
            s.append(I.replaceMin(FE.chordDictInv[seq[j][0]]))
    return s
 
# A  B  D    
# 0  10 20
def findChordIndexForTime(t, seq):
    for i in range(len(seq) - 1):
        if t < seq[i + 1][1]:
            return i
    return len(seq) - 1
    
def createVideo(seq, endtime):
    SEMI_60 = ImageFont.truetype("fonts/Montserrat-SemiBold.ttf", 60)
    SEMI_20 = ImageFont.truetype("fonts/Montserrat-SemiBold.ttf", 20)
    fps = 30
    outVideo = cv2.VideoWriter("chordVideo.mp4", cv2.VideoWriter_fourcc(*'DIVX'), fps, (600, 300))
    
    outFrame = np.zeros((300, 600, 3), dtype = np.uint8)
    number = 0
    positionPointer = 0
    while True:
        pointer = findChordIndexForTime(number / fps, seq)
        chordsStrings = getChordStrings(seq, pointer)
        
        black = Image.fromarray(outFrame)
        draw = ImageDraw.Draw(black)
        draw.text((50, 150), chordsStrings[0], font=SEMI_20, anchor="mm", stroke_width=0, fill=(255, 255, 255, 100))
        draw.text((175, 150), chordsStrings[1], font=SEMI_20, anchor="mm", stroke_width=0, fill=(255, 255, 255, 100))
        draw.text((300, 150), chordsStrings[2], font=SEMI_60, anchor="mm", stroke_width=0, fill=(255, 255, 255, 100))
        draw.text((425, 150), chordsStrings[3], font=SEMI_20, anchor="mm", stroke_width=0, fill=(255, 255, 255, 100))
        draw.text((550, 150), chordsStrings[4], font=SEMI_20, anchor="mm", stroke_width=0, fill=(255, 255, 255, 100))
        
        outVideo.write(np.array(black))
        number += 1
    
        if (number / fps) > endtime:
            break
    outVideo.release()
    
def combineAudioVideoAndOpen(title):
    subprocess.run(['ffmpeg', '-i', 'chordVideo.mp4', '-i', title, '-c', 'copy', 'output.mkv'])
    subprocess.run(['C:\\Program Files\\VideoLAN\\VLC\\vlc.exe', 'output.mkv', '--video-on-top'])
    
def main():
    title = "delilah.wav"
   
    x, Fs = librosa.load(title)
    oldInf, inference, times = I.netInference(x, Fs)
    
    assert(len(inference) == len(times))
    
    sequence = [(inference[0], times[0])] #Collapse consecutive chords of same label, only record changes
    for i in range(1, len(inference)):
        if inference[i] != sequence[-1][0]:
            sequence.append((inference[i], times[i]))
    
    createVideo(sequence, times[-1])
    combineAudioVideoAndOpen(title)
        
        
if __name__ == '__main__':
    main()
    