# The Clean RNN's 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/eubinecto/the-clean-rnns/main/run_deploy.py)

<p align="center">
  <img width="836" alt="image" src="https://user-images.githubusercontent.com/56193069/162101921-48ca93d2-787b-4eef-8a5b-00f31a3dba8c.png">
</p>


wandb와 pytorch-lightning으로 밑바닥부터 깔끔하게 구현해보는 RNN 👨‍👩‍👧‍👦: RNN, LSTM, BiLSTM 그리고 BiLSTM with attention mechanism.
## Who is this project for?
### 1️⃣ RNN 패밀리를 Pytorch로 밑바닥부터 구현해보고 싶다!
`cleanrnns/rnns.py`에 깔끔하게 구현해뒀어요 😊 

<details>
<summary> 예를 들면? </summary>
  
- [X] `RNNCell`, `RNN`
- [X] `LSTMCell`, `LSTM`
- [ ] `BiLSTMCell`, `BiLSTM`
- [ ] `BiLSTMSearchCell`, `BiLSTMSerach` (BiLSTM with attnetion mechanism)
  
</details>


### 2️⃣ 모델을 변경하면 정말로 성능이 오르는지 확인해보고 싶다!

모델만을 종속변인으로 두고 실험하는 것이 가능하도록 설계해뒀어요 📝

<details>
<summary> 예를 들면? </summary>
  
#### Naver Sentiment Movie Corpus 긍/부정 이진분류 성능 비교 
모델 | f1 score (train) | f1 score (validation) | f1 score (test) | 하이퍼파라미터 | wandb 로그
--- |------------------|------------| --- |--------| ---
RNN  | 0.8866           | 0.8457     | **0.8411** | 통제변인     | [학습](https://wandb.ai/eubinecto/the-clean-rnns/runs/40ca3shv?workspace=user-eubinecto) / [테스트](https://wandb.ai/eubinecto/the-clean-rnns/runs/20pfhypk/overview)
LSTM | 0.9184           | 0.8567     | **0.8522** | 통제변인     | [학습](https://wandb.ai/eubinecto/the-clean-rnns/runs/25wm1ome?workspace=user-eubinecto) / [테스트](https://wandb.ai/eubinecto/the-clean-rnns/runs/25e9xjyz/overview) 

동일한 문제에 대한 예측값도 [웹 데모](https://share.streamlit.io/eubinecto/the-clean-rnns/main/run_deploy.py)에서 비교가능해요 |
--- | 
<img width="748" alt="image" src="https://user-images.githubusercontent.com/56193069/162099283-ccb7dc8a-4a27-4954-af18-07498c3c7389.png"> |

  
</details>



### 2️⃣ 데이터 구축부터 모델 배포까지 모든 과정을 구현해보고 싶다!
단계별로 스크립트를 작성해뒀어요 🪜 

<details>
<summary> 예를 들면? </summary>
  
- [X] `run_build_nsmc.py` (데이터 구축)
- [X] `run_build_tokenizer.py` (토크나이저 구축) 
- [X] `run_train.py` (모델 훈련)
- [X] `run_test.py` (모델 평가) 
- [X] `run_deploy.py` (모델 배포)
- [ ] `run_tune.py` (하이퍼파라미터 튜닝)
  
</details>


### 3️⃣ 지저분한 노트북 튜토리얼보단 구조가 잡힌 자연어처리 프로젝트를 찾고있다!

객체지향, 함수지향 프로그래밍을 적재적소에 활용하여 `cleanrnns` 패키지를 정리해뒀어요 🧹 

<details>
<summary> 예를 들면? </summary>
  

- `datamodules.py` (객체지향 - 학습에 사용할 데이터셋을 `pl.LightningDataModule`객체로 추상화)
- `datasets.py` (객체지향 - 풀고자하는 문제에 따른 데이터의 형식을 `torch.utils.data.Dataset`객체로 추상화)
- `fetchers.py` (함수지향 - 데이터를 로드 및 다운로드하는 로직을 함수로 정의)
- `models.py` (객체지향 - 풀고자하는 문제의 형식을 `pl.LightningModule` 객체로 추상화)
- `paths.py` (로컬 경로 정의)
- `pipelines.py` (객체지향 - 예측에 필요한 로직을 하나의 객체로 추상화)
- `preprocess.py` (함수지향 - 데이터 전처리에 필요한 로직을 함수로 정의)
- `rnns.py`(객체지향 - 각 RNN 모델을 `torch.nn.Module`로 추상화)
- `tensors.py` (함수지향 - 데이터셋 -> `torch.Tensor` 변환에 필요한 로직을 함수로 정의)
  
</details>


