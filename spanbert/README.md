# SpanBERT(Improving Pre-training by Representing and Predicting Spans)
SpanBERT 구현 입니다.


## pretrain data
pretrain을 위한 데이터를 만드는 과정 입니다.

아래 명령을 실행하면 됩니다.

```sh
$ python data.py
```
- input: 입력 파일 입니다.. (기본 값: ../data/kowiki.json)
- output: 저장 파일 입니다. (기본 값: ../data/kowiki_span.json)
- n_seq: 최대 token 길이 입니다. (기본 값: 512)
- vocab: 사용할 vocab 파일 입니다. (기본 값: ../kowiki.model)
- mask_prob: mask 확률 입니다. 입니다. (기본 값: 0.15)


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
- epoch: 학습 epoch 입니다. (기본 값: 3)
- batch: 학습 batch_size 입니다. (기본 값: 28)
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
- vocab: 사용할 vocab 파일 입니다. (기본 값: ../kowiki.model)
- train: train 데이터 파일 입니다. (기본 값: ../data/ratings_train.json)
- test: test 데이터 파일 입니다. (기본 값: ../data/ratings_test.json)
- save: 모델을 저장할 위치 입니다. (기본 값: save_best.pth)
- pretrain: pretrain 된 모델 위치 입니다. (기본 값: save_pretrain.pth)
- epoch: 학습 epoch 입니다. (기본 값: 10)
- batch: 학습 batch_size 입니다. (기본 값: 128)
- gpu: 학습을 실행할 GPU 입니다. (기본 값: None) 특정 GPU에서만 동작하길 원하는 경우는 0, 1 과 같이 GPU ID를 지정해 주면 됩니다.
- seed: 랜덤 seed 입니다. (기본 값: 42)


## 결과
마지막 classfication 위한 값을 뽑는 과정을 cls token, output mean, output max 3가지로 확인해 봤습니다. 미세하지만 max가 좀더 좋은 효과를 냈습니다.

| ITEM                | Pretrain | logit | epoch  | loss  | accuracy |
|---------------------|----------|-------|--------|-------|----------|
| spanbert-pre:0-cls  | 0        | cls   | 9      | 0.220 | 0.825    |
| spabbert-pre:5-cls  | 5        | cls   | 9      | 0.201 | 0.826    |
| spabbert-pre:5-mean | 5        | mean  | 9      | 0.178 | 0.826    |
| spabbert-pre:5-max  | 5        | max   | 9      | 0.184 | 0.827    |
| spanbert-pre:10-max | 10       | max   | 5      | 0.254 | 0.827    |

#### loss
![](./img/loss.svg)

#### accuracy
![](./img/accuracy.svg)

