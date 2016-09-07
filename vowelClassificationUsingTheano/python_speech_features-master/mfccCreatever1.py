from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import numpy as N
import glob

import os
from os.path import isfile, join
#from os.path import osp

#(rate,sig) = wav.read("aTrain1.wav")
#mfcc_feat = mfcc(sig,rate)
#fbank_feat = logfbank(sig,rate)

#print (fbank_feat[1:3,:])
#print("printing all mfcc")
#print (mfcc_feat)
#print("printing shape or dimension of mfcc array")
#mfcc.shape)
#print(N.shape(mfcc_feat))
#print("printing type of numpy array mfcc")
#http://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.ndarray.html
#print(type((mfcc_feat)))

#import numpy as np
#x = N.arange(20).reshape((4,5))
#N.savetxt('test.txt', mfcc_feat)

#for x in range(1,5):
 #       stringname="aTrain%d"%x".wav"
	#(rate,sig) = wav.read("aTrain%d"%x".wav")
	#mfcc_feat = mfcc(sig,rate)
#	print("aTrain%d"%x)
 #       print(stringname)

for filename in glob.glob('*.wav'):
    print(filename)
    (rate,sig) = wav.read(filename)
    mfcc_feat = mfcc(sig,rate)
    print("printing shape or dimension of mfcc array")
    print(N.shape(mfcc_feat))
    print (mfcc_feat)
   # N.savetxt('%s.txt'%filename, mfcc_feat)

train_dir_a='/home/tauseef/gmm/annMustReadTexts/internetPythonCode/theano-tutorial-master/python_speech_features-master/train/a'
test_dir_a='/home/tauseef/gmm/annMustReadTexts/internetPythonCode/theano-tutorial-master/python_speech_features-master/test/a'
test_dir_e='/home/tauseef/gmm/annMustReadTexts/internetPythonCode/theano-tutorial-master/python_speech_features-master/test/e'

print("Collect all mfccs")
#files = [f for f in os.listdir(train_dir_a) if os.path.isfile(f) and f.endswith(".wav")]
files = [f for f in os.listdir(train_dir_a) if os.path.isfile(os.path.join(train_dir_a, f))  and f.endswith(".wav")]
print(files)
for j in files:
    print(j)

files2 = [f for f in os.listdir('./test') if os.path.isfile(f) and f.endswith(".wav")]
print(files2)
for j in files2:
    print(j)







#files3 = [f for f in os.walk(r_dir) if os.path.isfile(f) and f.endswith(".wav")]
print("All training files for a handling done now...")
#listTrainfilesA = [f for f in os.listdir(train_dir_a) if os.path.isfile(f) and f.endswith(".wav")]
listTrainfilesA = [f for f in os.listdir(train_dir_a) if os.path.isfile(os.path.join(train_dir_a, f))  and f.endswith(".wav")]

print(listTrainfilesA)

#for root in os.walk(train_dir_a):
for root, sub, files in os.walk(train_dir_a):
    #print(root)

    for j in listTrainfilesA:
        print(j)
        #(rate,sig) = wav.read(j)
        (rate,sig) = wav.read(os.path.join(root, j))

        mfcc_feat = mfcc(sig,rate)
        print("printing shape or dimension of mfcc array")
        print(N.shape(mfcc_feat))
        print (mfcc_feat)
        #N.savetxt('%s.txt'%j, mfcc_feat)
        #N.savetxt(train_dir,mfcc_feat,'%s.txt'%j)
    #N.savetxt('%s.txt'%j, mfcc_feat)
    #txtPath = osp.join(os.expanduser(train_dir));
    #print(txtPath)
        N.savetxt(r"/home/tauseef/gmm/annMustReadTexts/internetPythonCode/theano-tutorial-master/python_speech_features-master/train/a/%s.txt"%j,mfcc_feat)



train_dir_e='/home/tauseef/gmm/annMustReadTexts/internetPythonCode/theano-tutorial-master/python_speech_features-master/train/e'
print("All training files for e handling done now...")
#listTrainfilesE = [f for f in os.listdir(train_dir_e) if os.path.isfile(f) and f.endswith(".wav")]
listTrainfilesE = [f for f in os.listdir(train_dir_e) if os.path.isfile(os.path.join(train_dir_e, f))  and f.endswith(".wav")]

print(listTrainfilesE)

#for root in os.walk(train_dir_a):
for root, sub, files in os.walk(train_dir_e):
    #print(root)

    for j in listTrainfilesE:
        print(j)
        #(rate,sig) = wav.read(j)
        (rate,sig) = wav.read(os.path.join(root, j))

        mfcc_feat = mfcc(sig,rate)
        print("printing shape or dimension of mfcc array")
        print(N.shape(mfcc_feat))
        print (mfcc_feat)
        #N.savetxt('%s.txt'%j, mfcc_feat)
        #N.savetxt(train_dir,mfcc_feat,'%s.txt'%j)
    #N.savetxt('%s.txt'%j, mfcc_feat)
    #txtPath = osp.join(os.expanduser(train_dir));
    #print(txtPath)
        N.savetxt(r"/home/tauseef/gmm/annMustReadTexts/internetPythonCode/theano-tutorial-master/python_speech_features-master/train/e/%s.txt"%j,mfcc_feat)



#files3 = [f for f in os.walk(r_dir) if os.path.isfile(f) and f.endswith(".wav")]
print("testing files for a handling now...")
#listTrainfilesA = [f for f in os.listdir(train_dir_a) if os.path.isfile(f) and f.endswith(".wav")]
listTestfilesA = [f for f in os.listdir(test_dir_a) if os.path.isfile(os.path.join(test_dir_a, f))  and f.endswith(".wav")]

print(listTestfilesA)

#for root in os.walk(train_dir_a):
for root, sub, files in os.walk(test_dir_a):
    #print(root)

    for j in listTestfilesA:
        print(j)
        #(rate,sig) = wav.read(j)
        (rate,sig) = wav.read(os.path.join(root, j))

        mfcc_feat = mfcc(sig,rate)
        print("printing shape or dimension of mfcc array")
        print(N.shape(mfcc_feat))
        print (mfcc_feat)
        #N.savetxt('%s.txt'%j, mfcc_feat)
        #N.savetxt(train_dir,mfcc_feat,'%s.txt'%j)
    #N.savetxt('%s.txt'%j, mfcc_feat)
    #txtPath = osp.join(os.expanduser(train_dir));
    #print(txtPath)
        N.savetxt(r"/home/tauseef/gmm/annMustReadTexts/internetPythonCode/theano-tutorial-master/python_speech_features-master/test/a/%s.txt"%j,mfcc_feat)





#files3 = [f for f in os.walk(r_dir) if os.path.isfile(f) and f.endswith(".wav")]
print("testing files for a handling now...")
#listTrainfilesA = [f for f in os.listdir(train_dir_a) if os.path.isfile(f) and f.endswith(".wav")]
listTestfilesE = [f for f in os.listdir(test_dir_e) if os.path.isfile(os.path.join(test_dir_e, f))  and f.endswith(".wav")]

print(listTestfilesE)

#for root in os.walk(train_dir_a):
for root, sub, files in os.walk(test_dir_e):
    #print(root)

    for j in listTestfilesE:
        print(j)
        #(rate,sig) = wav.read(j)
        (rate,sig) = wav.read(os.path.join(root, j))

        mfcc_feat = mfcc(sig,rate)
        print("printing shape or dimension of mfcc array")
        print(N.shape(mfcc_feat))
        print (mfcc_feat)
        #N.savetxt('%s.txt'%j, mfcc_feat)
        #N.savetxt(train_dir,mfcc_feat,'%s.txt'%j)
    #N.savetxt('%s.txt'%j, mfcc_feat)
    #txtPath = osp.join(os.expanduser(train_dir));
    #print(txtPath)
        N.savetxt(r"/home/tauseef/gmm/annMustReadTexts/internetPythonCode/theano-tutorial-master/python_speech_features-master/test/e/%s.txt"%j,mfcc_feat)




for root, sub, testfiles in os.walk(test_dir_a):
    print(testfiles)
for j in testfiles:
    print(j)
    (rate,sig) = wav.read(os.path.join(root, j))
    mfcc_feat = mfcc(sig,rate)
    print("printing shape or dimension of mfcc array")
    print(N.shape(mfcc_feat))
    print (mfcc_feat)
    N.savetxt('%s.txt'%j, mfcc_feat)



for root, sub, testfiles in os.walk(test_dir_e):
    print(testfiles)
for j in testfiles:
    print(j)
    (rate,sig) = wav.read(os.path.join(root, j))
    mfcc_feat = mfcc(sig,rate)
    print("printing shape or dimension of mfcc array")
    print(N.shape(mfcc_feat))
    print (mfcc_feat)
    N.savetxt('%s.txt'%j, mfcc_feat)





