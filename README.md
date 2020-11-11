# 2020_Noise_Canceling_Project
 2020 NS2020팀 졸업작품 - Noise Canceling(Reduction) 프로젝트입니다. (남윤원, 이명신 진행)

## 목적 ##
* 딥러닝을 기반으로 한 Noise Canceling을 구현하며, 합성곱 신경망(CNN), 순환 신경망(RNN) 기법을 사용하고 두 방법을 비교하여 봅니다.
* 기존에 연구된 논문들과 오픈소스를 기반으로 하여 실생활에서 발생할 수 있는 여러 소음에 잘 대처할 수 있게 모델을 설계/구현하고 소프트웨어를 완성합니다.


## 데이터 정의 ##
1. Noise Data (총 8가지의 소리 파일)
 - Pink Noise
 - White Noise
 - Brown Noise
 - Children Playing
 - Drill
 - Engine
 - Jackhammer (망치 소리)
 - Air Conditional

2. Original Data (소음이 없는 깔끔한 소리)
 - MIR-1K DataSet (중국어로 노래부르는 소리) - 1000개
 - Korean Single Speaker Speech Dataset (한국어 음성 파일) - 12854개
 - 실생활에서 핸드폰/노트북 마이크 등으로 녹음한 소리 - 50개

3. Original Data와 Noise data를 섞어서 NoiseAdded 데이터를 만들어 학습 진행
(처음부터 Noise가 섞인 데이터를 수집하여 학습에 사용한다면, 모델을 훈련함에 있어 Noise의 Label 또는 Feature를 특정해 내기가 어려워 결과가 잘 나오지 않는 상황이 반복되었습니다.)

## 레퍼런스 ##

 다음의 오픈소스를 참고하여 제작하였습니다.

* CNN 기반의 Noise Reduction & Canceling OpenSource Code

https://github.com/daitan-innovation/cnn-audio-denoiser
 : 저희가 가장 많이 참고한 CNN 기반 오픈소스입니다. 이를 통해 어떻게 모델을 구축해야 하는지, 층을 어떻게 구성해야 하는지를 알수 있었습니다.

https://github.com/AP-Atul/Audio-Denoising
 : 저희 데이터로 실행해본 결과 효과가 있지 않았으며, 데이터의 문제라기 보다는 모델 자체의 문제로 생각되어 이 오픈소스는 활용하지 않았습니다.

https://github.com/timsainb/noisereduce
 : Noise를 Labeling하기 위해 FFT를 수행하였는데 이 아이디어를 알려준 오픈소스/자료입니다. 코드 자체는 C++로 구현되어 있으며 파이썬으로는 noisereduce라는 라이브러리로 제공되고 있었습니다. (https://pypi.org/project/noisereduce/#files)
졸업작품을 수행할 때 처음에는 이 라이브러리를 그대로 사용해 보면서, 저희가 수집한 데이터셋의 질을 검증해 볼 수 있었습니다.
데이터의 품질이 좋지 않은 경우 데이터셋을 재구성할 수 있었습니다.

https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise
 : Noise Reduction을 수행하는 다양한 논문이 제시된 사이트로 작업을 하면서 틈틈이 참고했던 문헌들입니다.


* RNN 기반의 Noise Reduction & Canceling OpenSource Code

https://github.com/adityatb/noise-reduction-using-rnn
 : LSTM 기반의 Noise Canceling을 구현해 놓은 오픈소스입니다. 하지만, 2018년에 구현된 소스이며 Tensorflow 1.2버전을 지원하기 때문에 현재 Tensorflow 2.0버전과 고수준의 Keras API를 이용할 수 없어 굉장히 불편하였고, Deprecated된 함수의 사용으로 인해 코드 자체가 동작하지 않았습니다. 때문에 저희는 이 소스를 Tensorflow 2.0버전으로 뜯어고쳐서 구현해 보았습니다.

https://github.com/ShinoharaYuuyoru/NoiseReductionUsingGRU
 : LSTM, GRU 기반의 Noise Canceling을 구현해 놓은 오픈소스입니다. 현재 RNN을 이용한 Noise Canceling의 경우 LSTM, GRU 기반을 넘어서 Transformer라는 방법이 대세가 되고 있다고 합니다. 하지만 저희의 경우 GRU를 이해하는 것이 많이 버거웠고, 이 오픈 소스 역시 동작시키는데 큰 애를 겪었으며, 결정적으로 저희가 이 오픈 소스를 수정/보완해서 더 나은 방법을 제시하기는 어렵다고 판단되어 사용하지 않았습니다. 다만, 어떻게 RNN 모델을 만들어야 하고, 데이터를 어떻게 구성하고 불러와서 학습을 시켜야 하며 Test는 어떻게 진행하는지를 알게 해 준 오픈소스였기에 첨부하였습니다. 저희가 구현한 RNN 소스 역시 TrainModel.py 파일과 TestModel.py 파일로 구성하였는데 이 오픈 소스의 설계에서 영감을 받았습니다.



* ... 기법을 사용하면 Audio Data의 Feature를 잘 추출할 수 있다.. -> 어떤 Features를 추출했는가? 그리고 이렇게 말할 수 있는 근거는 무엇인가? 또한 왜 이러한 이론적 배경을 이용하여 CNN, RNN 모델을 설계하고자 하였는가? 에 대해 이해한 대로 설명하겠습니다.

 Audio Data는 대부분 Nonstationary(시간에 독립적)입니다. 즉, Audio Data를 신호라고 한다면, 이 신호의 평균과 분산은 시간이 지남에 따라 변하게 됩니다. 
이 경우 전제 오디오 신호에 대한 푸리에 변환을 계산을 하는 것은 그다지 의미가 없게 됩니다.
그 대신, Audio를 작은 size로 쪼갠후 Fourier Transform을 수행하는 Short Term Fourier Transform를 이용하면, 긴 오디오 신호를 짧게 세분화하여 계산하기 때문에 각 시간마다 audio data의 feature를 추출할 수 있게 됩니다. 따라서 신호의 평균과 분산이 시간에 대해 계속 변해서 생기는 문제인 특정 시간의 오디오 신호가 무시될 수 있는 문제를 해결할 수 있다고 생각하였습니다.

 Audio Data를 Fourier Transform한 결과를 통해 우리가 알 수 있는 정보는, 예를 들면 신호의 주파수 성분이 10Hz, 20Hz, 30Hz로 이루어져 있다는 사실입니다. 그러나 3가지 주파수 성분이 어느 시점에 존재하는지 여부는 알기 어렵습니다.
 이는 복잡한 신호일수록 더더욱 알기 어려워지며, 이에 따라 기본 Time-Frequency 분석기법인 STFT(Short-Term Fourier Transfrom)가 생기게 되었습니다.
 STFT를 적용한 그래프는 시간과 주파수 2개의 축으로 이루어져 있기 때문에, STFT를 하게 되면 신호에 대해 자신이 알고 싶은 시점에서의 주파수 성분을 알 수 있다.
 이러한 processing을 거친 training data와 label의 격차를 줄이는 방식을 CNN 모델에서 활용하게 되었습니다.

 CNN 모델은 보통 이미지 Data에 유용하다고 알려져 있습니다. 하지만 audio data가 위에 명시한 Fourier Transform를 거쳐서 2차원 형태의 Data로 변환된다면, 이를 하나의 이미지로 바라볼 수 있기 때문에, 유용한 결과를 나타낼 것이라 예측하였습니다. 

 위의 오픈소스에서 사용된 DCNN (Deep Convolutional Neural Network)은 음성인식 향상을 위한 A Fully Convolutional Neural Network에서 수행 한 작업입니다.
 여기에서 CR-CED (Cascaded Redundant Convolutional Encoder-Decoder Network)가 제안되었는데, 이 모델은 대칭 인코더-디코더 아키텍처를 기반으로 합니다. 
 CR-CED 네트워크의 또 다른 중요한 특징은, convolution이 한 차원에서만 수행된다는 것이라고 합니다. 좀 더 구체적으로 말하면, 입력 스펙트럼 (129 x 8)이 주어지면 회선은 주파수 축 (즉, 첫 번째 축)에서만 수행됩니다. 이렇게 하면 전달 전파 중에 주파수 축이 일정하게 유지됩니다.

 두 구성 요소 모두 Convolution, ReLU 및 Batch Normalization의 반복 된 블록이 포함되어 있고, 전체적으로 네트워크에는 이러한 블록 16개가 포함되어 있었습니다.
 또한, 일부 인코더와 디코더 블록 사이에는 스킵 연결이 있으며, 여기서 두 구성 요소의 특징 벡터는 덧셈을 통해 결합된다.
 이는 ResNets와 매우 유사하여 스킵 연결은 수렴 속도를 높이고 그라디언트의 소실을 줄인다고 하였습니다.
 저희가 구현한 CNN 모델은 이를 기반으로 하여, 블록의 개수를 조절하고 Keras API로 모델을 재구성해 보았으며 여러 하이퍼파라미터(epoch 설정, learning_rate 설정 등..)를 조절해 보는 등의 절차를 거쳤습니다.

 RNN 모델은 과거의 Data를 학습에 다시 이용한다는 점에서 CNN과 차이를 보입니다. 그러나, 기존의 RNN Cell의 경우, 시간이 지날수록 오래된 데이터가 소실되는 문제가 있습니다. 또한 학습이 진행되면서 파라미터의 변화가 점차 미미해져서, 후반으로 갈수록 학습이 잘 되지 못하는 vanishing gradient 과 같은 문제가 있다. 따라서, 이러한 기존의 RNN 방식을 보완한 LSTM cell을 사용하여 보았습니다.

 입력값으로 STFT을 거친 (1024, 400) size의 vector를 입력받습니다. 모델 설계 과정에서, LSTM Cell의 개수를 512로 하여 진행한 경우 loss 가 0.6 e-5 까지 떨어지는 것을 확인할 수 있었습니다.다. 반면 bidirection 층을 쌓아서 진행한 경우 loss 2.2 e-5 로 떨어지며 값이 불안정한 모습을 보이는 걸로 보아 제대로 된 학습이 이루어지지 않음을 확인할 수 있었습니다. 그래서 512개의 Cell로 이루어진 모델을 최종적으로 채택하였습니다.


추후 업로드해야 할 사항
- CNN 모델 Training, Test 파일 최종 업로드
- 내용 보완, 모델 가중치 파일 업로드
- 샘플 sound 파일 업로드


보완해야 할 사항
* 잡음이 섞인 파일과 Noise Canceling된 최종 결과 파일의 파형을 비교하는 것으로 잡음이 줄었음을 증명 혹은 수치화해야 함

 직접 시연해볼 경우 그 차이를 느낄 수 있지만, RNN을 이용하여 구현한 모델에서는 제대로 Noise Canceling이 되지 못하는 경우가 다소 존재하며, 결정적으로 이를 수치화하기는 쉽지 않았습니다.
 따라서 저희는 Original Data의 파형 vs NoiseAdded Data의 파형 vs Noise Canceling된 Data의 파형 이렇게 3개를 비교해서, Original Data와 Noise Canceling된 Data의 파형을 비교해야 한다고 생각하였습니다. 
 이에 대해 한 가지 방법을 조언받았는데 바로 위 데이터를 각각 normalize하여 비교하고, accuracy를 측정하는 것이었습니다. 이 작업을 통해 객관적인 증명과 수치화가 필요합니다.

* 모델의 설계/구현을 왜 이렇게 했는지 이야기 할 때, 일부 모델의 case를 일반화할 수 있는가?

 그렇지 않다고 생각합니다. 그러나 현재 github에 올린 Final 파일은, 저희가 현재까지 테스트한 모델 중 가장 성능이 좋았던 모델입니다. 컴퓨팅 성능의 한계와 시간적인 비용 문제로 인해서, Google Colab Pro 버전을 이용하여 구글 클라우드 서비스를 통해, TPU 환경 하에서 모델 생성과 훈련을 진행하였습니다. 그 결과 기존의 로컬에서 학습을 진행하는 것에 비해 학습 속도를 많이 단축시킬 수 있었으며 그 결과 많은 테스트 결과를 얻어낼 수 있었습니다.

그러나 더욱 많은 모델을 만들어서 비교해 볼 필요가 있으며 때문에 계속 학습하여 보고, 비교하여 보아야 한다고 생각합니다. (수치화가 이루어질 경우, 어느 부분에 문제가 있음이 드러나면 이렇게 주먹구구식으로 무작정 찾아가는 것이 아닌, 방법론적으로 접근하여 더 나은 결과를 얻어낼 수 있지 않을까라는 생각 역시 하고 있습니다만, RNN 모델의 성능을 향상시킬 뚜렷한 해법을 현재 찾아내지는 못했습니다.)
