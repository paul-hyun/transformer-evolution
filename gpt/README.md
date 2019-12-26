# GPT(Generative Pre-Training)
GPT 구현 입니다.


## pretrain data
pretrain을 위한 데이터를 만드는 과정 입니다.

아래 명령을 실행하면 됩니다.

```sh
$ python data.py
```
- input: 입력 파일 입니다.. (기본 값: ../data/kowiki.json)
- output: 저장 파일 입니다. (기본 값: ../data/kowiki_gpt.json)
- n_seq: 최대 token 길이 입니다. (기본 값: 256)


## pretrain
pretrain 학습을 하는 과정 입니다.

아래 명령을 실행 하시면 됩니다.

```sh
$ python pretrain.py
```
- config: 설정 파일을 선택 합니다. (기본 값: config_half.json) 큰 파라미터를 사용하려면 config.json으로 변경 하세요.
- vocab: 사용할 vocab 파일 입니다. (기본 값: ../kowiki.model)
- input: 학습 데이터 파일 입니다. (기본 값: ../data/kowiki_gpt.json)
- save: 모델을 저장할 위치 입니다. (기본 값: save_pretrain.pth)
- epoch: 학습 epoch 입니다. (기본 값: 20)
- batch: 학습 batch_size 입니다. (기본 값: 256)
- gpu: 학습을 실행할 GPU 입니다. (기본 값: None) 특정 GPU에서만 동작하길 원하는 경우는 0, 1 과 같이 GPU ID를 지정해 주면 됩니다.
- seed: 랜덤 seed 입니다. (기본 값: 42)


## train
네이버 영화 데이터를 학습을 하는 과정 입니다.

아래 명령을 실행 하시면 됩니다.

```sh
$ python train.py
```
주요 옵션은 다음과 같습니다.
- config: 설정 파일을 선택 합니다. (기본 값: config_half.json) 큰 파라미터를 사용하려면 config.json으로 변경 하세요.
- lm: lm을 학습할 비율 입니다. (기본 값: 0.0) lm을 동시에 학습하기 원하면 0 과 1 사이의 값을 주면 됩니다.
- vocab: 사용할 vocab 파일 입니다. (기본 값: ../kowiki.model)
- train: train 데이터 파일 입니다. (기본 값: ../data/ratings_train.json)
- test: test 데이터 파일 입니다. (기본 값: ../data/ratings_test.json)
- save: 모델을 저장할 위치 입니다. (기본 값: save_best.pth)
- pretrain: pretrain 된 모델 위치 입니다. (기본 값: save_pretrain.pth)
- epoch: 학습 epoch 입니다. (기본 값: 20)
- batch: 학습 batch_size 입니다. (기본 값: 512)
- gpu: 학습을 실행할 GPU 입니다. (기본 값: None) 특정 GPU에서만 동작하길 원하는 경우는 0, 1 과 같이 GPU ID를 지정해 주면 됩니다.
- seed: 랜덤 seed 입니다. (기본 값: 42)


## 결과
- pretrain 60회 후 학습한 경우가 가장 좋은 결과를 냈습니다. (gpt-pre:60-lm:0)
- pretrain 횟수가 많아 질 수록 좋은 성능을 냈습니다.
- 더 많은 데이터를 더 많은 회수를 학습하면 더 좋은 결과를 낼 것으로 예상 됩니다.

| ITEM             | Pretrain | LM  | epoch  | loss  | accuracy |
|------------------|----------|-----|--------|-------|----------|
| gpt-pre:0-lm:0   | 0        | 0   | 15     | 0.347 | 0.820    |
| gpt-pre:20-lm:0  | 20       | 0   | 18     | 0.338 | 0.818    |
| gpt-pre:40-lm:0  | 40       | 0   | 19     | 0.325 | 0.833    |
| gpt-pre:60-lm:0  | 60       | 0   | 16     | 0.322 | 0.837    |

#### loss
![](./img/loss.svg)

#### accuracy
![](./img/accuracy.svg)
