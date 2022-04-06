# the-clean-rnns

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/eubinecto/the-clean-rnns/main/run_deploy.py)

wandb와 pytorch-lightning으로 밑바닥부터 깔끔하게 구현해보는 RNN 패밀리 👨‍👩‍👧‍👦

## Who is this project for?
### 1️⃣ RNN 패밀리를 Pytorch로 밑바닥부터 구현해보고 싶다!
`cleanrnns/rnns.py`에 구현해놨어요 👍 
  - [X] `RNNCell`, `RNN`
  - [ ] `LSTMCell`, `LSTM`
  - [ ] `BiLSTMCell`, `BiLSTM`
  - [ ] `BiLSTMSearchCell`, `BiLSTMSerach` (BiLSTM with attnetion mechanism)
### 2️⃣ 기왕하는거 데이터 구축부터 모델 배포까지 모든 과정을 구현해보고 싶다!
단계별로 스크립트를 구현해놨어요 👍 
  - [X] `run_build_nsmc.py` (데이터 구축)
  - [X] `run_build_tokenizer.py` (토크나이저 구축) 
  - [X] `run_train.py` (모델 훈련)
  - [X] `run_test.py` (모델 평가) 
  - [X] `run_deploy.py` (모델 배포)
  - [ ] `run_tune.py` (하이퍼파라미터 튜닝)
### 3️⃣ 지저분한 노트북 튜토리얼보단 구조가 잡힌 자연어처리 프로젝트의 템플릿을 찾고 있다!

객체지향, 함수지향 프로그래밍을 적재적소에 활용하여 `cleanrnns` 패키지를 정리해뒀어요 👍
  - `datamodules.py` (객체지향 - 학습에 사용할 데이터셋을 `pl.LightningDataModule`객체로 추상화)
  - `datasets.py` (객체지향 - 풀고자하는 문제에 따른 데이터의 형식을 `torch.utils.data.Dataset`객체로 추상화)
  - `fetchers.py` (함수지향 - 데이터를 로드 및 다운로드하는 로직을 함수로 정의)
  - `models.py` (객체지향 - 풀고자하는 문제의 형식을 `pl.LightningModule` 객체로 추상화)
  - `paths.py` (로컬 경로 정의)
  - `pipelines.py` (객체지향 - 예측에 필요한 로직을 하나의 객체로 추상화)
  - `preprocess.py` (함수지향 - 데이터 전처리에 필요한 로직을 함수로 정의)
  - `rnns.py`(객체지향 - 각 RNN 모델을 `torch.nn.Module`로 추상화)
  - `tensors.py` (함수지향 - 데이터셋 -> `torch.Tensor` 변환에 필요한 로직을 함수로 정의)
