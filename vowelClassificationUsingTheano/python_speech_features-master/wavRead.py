import glob
from scipy.io.wavfile import read
import numpy as np

wavs = []
g=len(wavs)
for filename in glob.glob('*.wav'):
    print(filename)
    wavs.append(read(filename))

print(wavs)
for i in range(0,g):
    h=wavs[i]
    g=h[1]
    np.savetxt('test-%d.dat' % i, g)
