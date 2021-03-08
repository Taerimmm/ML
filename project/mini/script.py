import numpy as np
import librosa
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

test_music = './project/mini/data/country.6.mp3'
y, sr = librosa.load(test_music)
print(y)
print(y.shape) # 661500


mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

print(mel_spect)
print(mel_spect.shape) # (128, 646) <- 646 = y // hop_length = 661500 // 1024


'''
n_fft : int > 0 [scalar]

    length of the windowed signal after padding with zeros.  
    The number of rows in the STFT matrix `D` is `(1 + n_fft/2)`.  
    The default value, `n_fft=2048` samples, corresponds to a physical  
    duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the  
    default sample rate in librosa. This value is well adapted for music  
    signals. However, in speech processing, the recommended value is 512,  
    corresponding to 23 milliseconds at a sample rate of 22050 Hz.  
    In any case, we recommend setting `n_fft` to a power of two for  
    optimizing the speed of the fast Fourier transform (FFT) algorithm.  
hop_length : int > 0 [scalar]

    number of audio samples between adjacent STFT columns.  

    Smaller values increase the number of columns in `D` without  
    affecting the frequency resolution of the STFT.  

    If unspecified, defaults to `win_length // 4` (see below).  
'''
power_to_db : converting to decibals
ref=np.max : Compute dB relative to peak power
dB는 자연계의 물리량 변화 비율을 표현하기 편리하고 간단한데다가 지수형태의 표기이기 때문에 
수의 곱을 합으로 표현할 수 있는 수리적 장점도 있어 과학과 엔지니어링 분야에 널리 쓰입니다.

mel_spectrogram 설명
- 오디오 신호는 여러 개의 단일 주파수 음파로 구성
- 푸리에 변환은 우리가 각각의 주파수의 주파수의 진폭으로 신호를 분해 할 수있는 수학 식
- 이는 모든 신호가 원래 신호에 합산되는 사인파 및 코사인 파 세트로 분해 될 수 있기 때문에 가능 (푸리에 정리)
- 고속 푸리에 변환(FFT)은 신호의 주파수 성분을 분석 할 수있는 강력한 도구
- 스펙트로그램은 서로의 위에 쌓인 FFT 묶음
- 신호의 크기 또는 진폭을 시각적으로 표현하는 방법
- 시간이 지남에 따라 다른 주파수에서 달라지기 때문
- MEL SPECTROGRAM은 주파수가 멜 스케일로 변환되는 스펙트로 그램


ResNet 설명
- 어느 일정 정도 이상의 layer 수를 넘어서게 되면, gradient vanishing 문제 발생
- gradient vanishing
  : backpropagation을 해도 앞의 layer일수록 미분값이 작아져 
   그만큼 output에 영향을 끼치는 weight 정도가 작아지는 것
- backpropagationArtificial (역전파)
  : Neural Network를 학습시키기 위한 일반적인 알고리즘 중 하나
    내가 뽑고자 하는 target값과 실제 모델이 계산한 output이 얼마나 차이가 나는지 구한 후 
    그 오차값을 다시 뒤로 전파해가면서 각 노드가 가지고 있는 변수들을 갱신하는 알고리즘
- 점선인 것은 feature map size가 반으로 줄어든 경우를 의미
- 점선의 shortcut connection의 경우 1x1 convolution과 batch normalization을 적용하기 때문에 학습 대상인 connectio

- 점선의 경우 dimension이 일치하지 않기 때문에(stride=(2,2) 이기 떄문에) dimension을 증가시켜 연산
- dimension을 증가시키기 위한 방법으로 2가지를 고려해볼 수 있음
 -> 1x1 convolution을 이용하여 dimension을 맞춰줌.

BatchNormalization
- 각각의 스칼라 Feature들을 독립적으로 정규화하는 방식. 
  즉, 각각의 Feature들의 Mean 및 Variance를 과 로 정규화를 하는 것
- batchnorm also maintains non-trainable weights, 
  which are updated via layer updates (i.e. not through backprop): the mean and variance vectors.