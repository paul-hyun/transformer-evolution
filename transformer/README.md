# Transformer
Transformer 구현 입니다.


## train
아래 명령을 실행 하시면 됩니다.
```sh
$ python train.py
```
주요 옵션은 다음과 같습니다.
- config: 설정 파일을 선택 합니다. (기본 값: config_half.json) 큰 파라미터를 사용하려면 config.json으로 변경 하세요.
- vocab: 사용할 vocab 파일 입니다. (기본 값: ../kowiki.model)
- save: 모델을 저장할 위치 입니다. (기본 값: save_best.pth)
- epoch: 학습 epoch 입니다. (기본 값: 10)
- batch: 학습 batch_size 입니다. (기본 값: 256)
- gpu: 학습을 실행할 GPU 입니다. (기본 값: None) 특정 GPU에서만 동작하길 원하는 경우는 GPU ID(0, 1, ...)를 지정해 주면 됩니다.
- seed: 랜덤 seed 입니다. (기본 값: 42)


## 결과
| ITEM              | Pretrain | epoch  | loss   | accuracy |
|-------------------|----------|--------|--------|----------|
| transformer-pre:0 | 0        | 19     | 0.3054 | 0.8312   |

#### loss
![](./img/loss.svg)

#### accuracy
![](./img/accuracy.svg)

