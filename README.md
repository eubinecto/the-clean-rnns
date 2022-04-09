# The Clean RNN's 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/eubinecto/the-clean-rnns/main/run_deploy.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ThRBOZYjJNZmOcs173qZroyaotNe7tSc?usp=sharing)
 <a href="https://wandb.ai/eubinecto/the-clean-rnns/artifacts"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg" height=20></a>
 [![Total alerts](https://img.shields.io/lgtm/alerts/g/eubinecto/the-clean-rnns.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/eubinecto/the-clean-rnns/alerts/)
 [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/eubinecto/the-clean-rnns.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/eubinecto/the-clean-rnns/context:python)

<p align="center">
  <img width="836" alt="image" src="https://user-images.githubusercontent.com/56193069/162101921-48ca93d2-787b-4eef-8a5b-00f31a3dba8c.png">
</p>


[wandb](https://wandb.ai/site)와 [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/)으로 깔끔하게 구현해보는 RNN 패밀리 👨‍👩‍👧‍👦


## Shortcuts
[`RNNCell`](https://github.com/eubinecto/the-clean-rnns/blob/0e30c8035f9ea29bd96edc23e8a8f9b8457a8a3c/cleanrnns/rnns.py#L24-L45) / [`RNN`](https://github.com/eubinecto/the-clean-rnns/blob/0e30c8035f9ea29bd96edc23e8a8f9b8457a8a3c/cleanrnns/rnns.py#L48-L56) / [`LSTMCell`](https://github.com/eubinecto/the-clean-rnns/blob/0e30c8035f9ea29bd96edc23e8a8f9b8457a8a3c/cleanrnns/rnns.py#L59-L89) / [`LSTM`](https://github.com/eubinecto/the-clean-rnns/blob/0e30c8035f9ea29bd96edc23e8a8f9b8457a8a3c/cleanrnns/rnns.py#L92-L98) / [`BiLSTMCell`](https://github.com/eubinecto/the-clean-rnns/blob/e718b0ae556702b3ca14e6b423afecd62a91f845/cleanrnns/rnns.py#L110-L122) /  [`BiLSTM`](https://github.com/eubinecto/the-clean-rnns/blob/e718b0ae556702b3ca14e6b423afecd62a91f845/cleanrnns/rnns.py#L125-L132)

## Who is this project for?
### 1️⃣ RNN 패밀리를 Pytorch로 밑바닥부터 구현해보고 싶다!
`cleanrnns/rnns.py`에 깔끔하게 구현해뒀어요 😊 

<details>
<summary> 예를 들면? </summary>
  
- [X] `RNNCell`, `RNN`
- [X] `LSTMCell`, `LSTM`
- [X] `BiLSTMCell`, `BiLSTM`
- [ ]  🚧 `GRUCell`, `GRU` 🚧
  
</details>


### 2️⃣ 모델을 변경하면 정말로 성능이 오르는지 확인해보고 싶다!

모델만을 종속변인으로 두고 실험하는 것이 가능하도록 설계해뒀어요 📝

<details>
<summary> 예를 들면? </summary>
  
#### Naver Sentiment Movie Corpus 긍/부정 이진분류 성능 비교 
모델  | f1 score (test) | 가중치 | 소요시간 | `hidden_size` | 하이퍼파라미터 |  wandb 로그
--- | --- |--------| --- | ---| --- | --- 
RNN  | 0.8411 | 16.4M | 18m 19s | 512 | 통일 | [학습](https://wandb.ai/eubinecto/the-clean-rnns/runs/40ca3shv?workspace=user-eubinecto) / [테스트](https://wandb.ai/eubinecto/the-clean-rnns/runs/20pfhypk/overview)
LSTM |  0.8522 | 16.4M | 20m 18s | 443 | 통일 |  [학습](https://wandb.ai/eubinecto/the-clean-rnns/runs/3eilxpo4/overview) / [테스트](https://wandb.ai/eubinecto/the-clean-rnns/runs/2vimv04k/overview) 
BiLSTM | **0.8539** | 16.4M | **36m 12s** | 387 | 통일 |  [학습](https://wandb.ai/eubinecto/the-clean-rnns/runs/cyos30w7/artifacts) / [테스트](https://wandb.ai/eubinecto/the-clean-rnns/runs/38zie0fu/overview)

 동일한 입력에 대한 예측값도 [웹 데모](https://share.streamlit.io/eubinecto/the-clean-rnns/main/run_deploy.py) 에서 한눈에 비교가 가능해요 |
 --- | 
  <img width="350" alt="image" src="https://user-images.githubusercontent.com/56193069/162197935-4eddc14d-8580-48f9-82ab-64e97d2f877f.png"> |
 

</details>

### 3️⃣ 데이터 구축부터 모델 배포까지 모든 과정을 구현해보고 싶다!
단계별로 스크립트를 작성해뒀어요 🪜 

<details>
<summary> 예를 들면? </summary>
  
- [X] `run_build_nsmc.py` (데이터 구축)
- [X] `run_build_tokenizer.py` (토크나이저 구축) 
- [X] `run_train.py` (모델 훈련)
- [X] `run_test.py` (모델 평가) 
- [X] `run_deploy.py` (모델 배포)
- [ ] 🚧 `run_tune.py` (하이퍼파라미터 튜닝)  🚧
  
</details>


### 4️⃣ 지저분한 노트북 튜토리얼보단 구조가 잡힌 자연어처리 프로젝트를 찾고있다!

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


## Quick Start 

데이터 구축부터 모델 평가까지 진행해볼 수 있는 [Colab 노트북](https://colab.research.google.com/drive/1WIPOP5_xGHCKK4g8r9GjNiY_pLo5PA4e?usp=sharing)을 만들어뒀어요. 실행을 해보면서 궁금한 점이 있다면 이슈를 남겨주세요 😊

### 프로젝트 설치 및 환경설정

```shell
git clone https://github.com/eubinecto/the-clean-rnns.git  # 프로젝트 클론
cd the-clean-rnns  # 루트 디렉토리 설정
pip3 install -r requirements.txt  # 의존 라이브러리 설치 
wandb login  # Weights & Biases 계정 로그인 (회원가입 필요)
```

### 데이터 전처리 및 구축 

```shell
python3 run_build_nsmc.py  # Naver Sentiment Movie Corpus
```

`fetch_nsmc`로 구축한 데이터를 간편하게 확인해볼 수 있습니다. Korpora에서 제공하는 nsmc는 기본적으로 validation셋이 없지만,
[전처리](https://github.com/eubinecto/the-clean-rnns/blob/f7e14c53920fe21c333d301c91f5a1d5b0501bb1/run_build_nsmc.py#L19)를 통해 구축합니다.
validation셋의 비율과, 랜덤 스플릿 시드는[ `config.yaml`](https://github.com/eubinecto/the-clean-rnns/blob/f7e14c53920fe21c333d301c91f5a1d5b0501bb1/config.yaml#L4-L5)에서 설정가능합니다.
```python
from cleanrnns.fetchers import fetch_nsmc
train, val, test = fetch_nsmc()
for row in train.data[:10]:
    print(row[0], row[1])
```

### 토크나이저 구축

```shell
python3 run_build_tokenizer.py # BPE
```

토크나이저도 `fetch_tokenizer`로 간편하게 확인해볼 수 있습니다. 본 프로젝트에서는 Byte Pair Encoding 알고리즘으로 구축하며, 어휘의 크기와 스페셜토큰은 [`config.yaml`](https://github.com/eubinecto/the-clean-rnns/blob/f7e14c53920fe21c333d301c91f5a1d5b0501bb1/config.yaml#L7-L15) 에서 설정 가능합니다.

```python3
from cleanrnns.fetchers import fetch_tokenizer
tokenizer = fetch_tokenizer()
# 토크나이징
encoding = tokenizer.encode("이 영화 진짜 재미있다")
print(encoding.ids)
print([tokenizer.id_to_token(token_id) for token_id in encoding.ids])
# 스페셜 토큰
print(tokenizer.pad_token)  
print(tokenizer.token_to_id(tokenizer.pad_token))  
print(tokenizer.unk_token)  
print(tokenizer.token_to_id(tokenizer.unk_token))  
print(tokenizer.bos_token)  
print(tokenizer.token_to_id(tokenizer.bos_token)) 
print(tokenizer.eos_token)  
print(tokenizer.token_to_id(tokenizer.eos_token))  
#어휘의 크기
print(tokenizer.get_vocab_size())
```

데이터셋과 토크나이저를 구축한뒤에는, 텍스트 데이터를 학습에 사용하기 위해 `torch.Tensor` 객체로 변환해야합니다. 이를 위한 로직은
[`NSNC`](https://github.com/eubinecto/the-clean-rnns/blob/f7e14c53920fe21c333d301c91f5a1d5b0501bb1/cleanrnns/datamodules.py#L17-L65) 클래스 (`LightningDataModule`)에 담겨 있으며, 모델학습을 진행할 때 내부적으로 사용됩니다. 물론 아래와 같이 텐서변환과정을 
간단하게 확인해볼수는 있습니다.
```python3
import os
from cleanrnns.fetchers import fetch_config
from cleanrnns.datamodules import NSMC
config = fetch_config()['rnn_for_classification']
config['num_workers'] = os.cpu_count()
tokenizer = fetch_tokenizer()
datamodule = NSMC(config, tokenizer)
datamodule.prepare_data()  # wandb로부터 구축된 텍스트데이터 다운로드
datamodule.setup()  # 텐서로 변환
print("--- A batch from the training set ---")
for batch in datamodule.train_dataloader():
    x, y = batch
    print(x)  # (N, L)
    print(x.shape)
    print(y)  # (N,)
    print(y.shape)
    break

print("--- A batch from the validation set ---")
for batch in datamodule.val_dataloader():
    x, y = batch
    print(x)  # (N, L)
    print(x.shape)
    print(y)  # (N,)
    print(y.shape)
    break

```

### 모델 훈련

```shell
python3 run_train.py rnn_for_classification
python3 run_train.py lstm_for_classification
python3 run_train.py bilsm_for_classification
```
RNN, LSTM, BiLSTM을 구축한 데이터에 학습시킵니다. `hidden_size`, `max_epochs`, 등의 하이퍼파라미터는 [`config.yaml`](https://github.com/eubinecto/the-clean-rnns/blob/f7e14c53920fe21c333d301c91f5a1d5b0501bb1/config.yaml#L18-L52)에서 설정가능합니다.

### 모델 평가
```shell
python3 run_test.py rnn_for_classification
python3 run_test.py lstm_for_classification
python3 run_test.py bilstm_for_classification
```
RNN, LSTM, BiLSTM의 성능을 구축한 테스트셋으로 측정합니다. 

### 모델 배포 (로컬)

```shell
streamlit run run_deploy.py
```
웹에 배포를 원하신다면 [Streamlit Cloud](https://streamlit.io/cloud) 사용을 추천!

## To-do's
- [ ] `ClassificationWithAttentionBase` -> `RNNForClassificationWithAttention`, `LSTMForClassificationWithAttention`, `BiLSTMForClassificationWithAttention`
- [ ] seq2seq 지원
- [ ] ner 지원

