# 사용되는 feature 탐구해보기!!

'''
librosa.feature.zero_crossing_rate
- 신호가 양수에서 0으로, 또는 음에서 0에서 양으로 변경되는 속도
- 음성 인식과 음악 정보 검색에 널리 사용되어 타악 음을 분류하는 핵심 기능
librosa.feature.spectral_centroid
- 스펙트럼을 특성화하기 위해 디지털 신호 처리에 사용되는 측정
- 스펙트럼의 질량 중심이있는 위치를 나타냅니다
librosa.feature.spectral_rolloff
- 총 스펙트럼 에너지의 지정된 백분율 (예 : 85 %)이 아래에있는 주파수
librosa.feature.mfcc
- '음성데이터'를 '특징벡터' (Feature) 화 해주는 알고리즘
- mel spectrogram을 행렬을 압축해서 표현해주는 DCT(Discrete Cosine Transform) 처리하면 얻게되는 coefficient
- mel scale로 변환한 스펙트로그램을 더 적은 값들로 압축하는 과정
- MFCC를 통해 음성 데이터를 분석하게 되면, 음원보다 상대적으로 갯수가 적은 coefficient들을 학습함으로써 보다 효율적으로 분석을 할 수 있다.
'''

'''
https://librosa.org/doc/main/feature.html 
여기 나와있는 추가적인 feature 몇가지 알아보기?

mfcc 제외한 나머지 것들은 mean을 사용하였는데 그 이유?
'''


import numpy as np
import librosa
import librosa.display

y, sr = librosa.load('../data/project_data/mini/genre/blues/blues.00000.wav')
# y : audio time series
# sr : sampling rate ; 이산적인 신호를 만들기 위해 연속적 신호에서 얻어진 단위시간(주로 초)당 샘플링 횟수
#                      주파수 분석을 하면 초당 1000개로 표현할 수 있게 됩니다 <- check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

''' ZCR (zero crossing rate)'''
zcr = librosa.feature.zero_crossing_rate(y)
# the rate of sign-changes of the signal during the frame
# ZCR can be interpreted as a measure of the noisiness of a signal
# it usually exhibits higher values in the case of noisy signals
# the values of ZCR are higher for the noisy parts of the signal, while in speech frames the respective ZCR values are generally lower
np.mean(zcr)


''' Spectral Centroid '''
spec_centroid = librosa.feature.spectral_centroid(y)
# Because the spectral centroid is a good predictor of the "brightness" of a sound
# it is widely used in digital audio and music processing as an automatic measure of musical timbre.
np.mean(spec_centroid)


''' Spectral Rolloff '''
spec_rolloff = librosa.feature.spectral_rolloff(y)
# the frequency below which a specified percentage of the total spectral energy (default : 0.85)
# The roll-off frequency can be used to distinguish between harmonic (below roll-off) and noisy sounds (above roll-off).
np.mean(spec_rolloff)


''' MFCC (Mel-Frequency Cepstral Coefficient) '''
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
np.mean(mfcc.T, axis=0)





# 이외 Feature 들