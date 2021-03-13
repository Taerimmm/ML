import numpy as np
import matplotlib.pyplot as plt
import librosa 
import librosa.display

import warnings
warnings.filterwarnings('ignore')

y, sr = librosa.load('../data/project_data/mini/genre/blues/blues.00000.wav')

print(y.shape)
print(sr)

plt.plot(y)
plt.title('Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.show()


# fft - fast Fourier transform
n_fft = 2048
ft = np.abs(librosa.stft(y[:n_fft], hop_length = n_fft+1))

plt.plot(ft)
plt.title('spectrum')
plt.xlabel('Frequency Bin')
plt.ylabel('Amplitude')
plt.show()


# Computing the spectrogram
spec = np.abs(librosa.stft(y, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max) # converting to decibals

# Plotting the spectrogram
plt.figure(figsize=(8,5))
librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()


# Computing the mel spectrogram
spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
spect = librosa.power_to_db(spect, ref=np.max) # Converting to decibals
print(spect)
print(type(spect))
print(spect.shape)

# Plotting the mel spectrogram
plt.figure(figsize=(8,5))
librosa.display.specshow(spect, y_axis='mel', fmax=8000, x_axis='time')
plt.title('Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.show()


''' Mel Frequency Cepstral Coefficients (MFCC) '''

# Extracting mfccs from the audio signal
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)

# Displaying the mfccs
plt.figure(figsize=(8,5))
librosa.display.specshow(mfcc, x_axis='time')
plt.title('MFCC')
plt.show()

mfccscaled = np.mean(mfcc.T, axis=0)
print(mfccscaled)
