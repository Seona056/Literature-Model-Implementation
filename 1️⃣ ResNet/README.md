# 목차

1. [[Implementation] ResNet Ablation Study ](#implementation-resnet-ablation-study)  
   - [개선 사항 1](#개선-사항-1)
   - [개선 사항 2](#개선-사항-2)
   - [개선 사항 3](#개선-사항-3)
 
2. [[Implementation] ResNet CIFAR-10 and Analysis ](#implementation-resnet-cifar-10-and-analysis)  


<br>

---
# [Implementation] ResNet Ablation Study  

<br>

> ✅ 기존 [AIFFEL GOINGE DEPPER [CV-02] 프로젝트](https://github.com/Seona056/AIFFEL_Daegu/blob/main/GOING%20DEEPER/%5BCV-02%5D%20ResNet%20Ablation%20Study.ipynb)의 코드를 필사 및 수정하였다.<br><br>
❗ 랜더링이 심하게 일어난다면 해당 깃허브의 <u>*README*</u>에 있는 [링크](https://nbviewer.org/github/Seona056/AIFFEL_Daegu/blob/main/GOING%20DEEPER/%5BCV-02%5D%20ResNet%20Ablation%20Study.ipynb)를 이용할 것 ❗

<br>

## 개선 사항 1 

기본 `conv_block`에서 BatchNorm layer에 momentum, epsilon울 추가하였다.

<br>

> 🔑**Batch Norm Layer**에서 **epsilon**이란❓<br><br>
- `momentum`: Momentum for the moving average.
- `epsilon` : Small float added to variance to avoid dividing by zero.  

<br>

- [keras 공식 문서 참고](https://keras.io/api/layers/normalization_layers/batch_normalization/)


<br>

## 개선 사항 2

<br>

**2-1) 반복문 내의 if문의 코드를 수정** : `resblock()` 및 `build_resnet()`의 코드 수정

<br>

**2-2)`build_resnet()` 코드 수정**
- ***GAP***, ***FC-layer***, ***model*** 코드를 <u>for문 밖으로 꺼냄</u>
- ***name*** 을 따로 if문을 만드는 대신, ***regularizer***를 구분하기 위해 생성한 if문안으로 넣어줌
- ***`conv_1`*** 블럭에 들어가기 전, **MaxPooling**의 filter size를 ***`(2,2)`*** 👉 ***`(3,3)`*** 로 변경 (논문에 따름)  

<br>

## 개선 사항 3

<br>

아래의 코드를 추가 하여, 모델의 구조를 `plot_model`로 확인 할 수 있도록 함.  
```
from tensorflow.keras.utils import plot_model

plot_model(모델명)
```
 
<br><br>

---
# [Implementation] ResNet CIFAR-10 and Analysis

<br>

논문에서는 `for ImageNet` ResNet 뿐만 아니라 ***`📊 for CIFAR-10 and Analysis`*** 모델 아키텍쳐를 따로 기술해 두었다.

<br>

[**Model Architectures**](https://www.notion-pinotnoir056.com/75a919d0-a498-4264-a5f7-375a3c77649f#8096143b-61ef-42a9-9bdf-b9937d0a3d6b)

- input : 32 x 32 images (픽셀 당 평균은 차감)
- 첫 번째 레이어 **3 x 3 conv**
- 이후 6n개의 3 x 3 conv 를 쌓는다. 
- 각 **`{32, 16, 8}`** 사이즈의 특징 맵
- 특징 맵 마다 ***2n*** 레이어를 가진다.
- **`n= {3, 5, 7, 9}`** ( 22, 32, 44, 56-layer networks)
- 필터 갯수는 각 **`{16, 32, 64}`**개 이다.
- Stride 2
- GAP, 10-way FC layer (softmax)
- shortcut은 the pair of 3 x 3 layers에 연결된다. (총 3n shortcuts)

![](https://oopy.lazyrockets.com/api/v2/notion/image?src=https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F3ab9aed9-1298-4181-ba88-db05b6dbff3f%2FUntitled.png&blockId=12954056-51d9-44f5-9225-ade37fe8ccf3)

- 각 면에 4개의 픽셀이 패딩되어 있으며 패딩된 이미지 또는 수평 플립에서 32×32의 크롭이 무작위로 샘플링된다.
- Batch Normalization (BN) : 매 conv 직후, 활성 함수 이전
- 가중치 초기화 : He initialization
- Train : 처음부터 plain / residual nets로 학습
- SGD, mini-batch size of 128, 2개의 GPU
- Learning Rate :  0.1부터 시작  32k, 48k iterations 마다 10으로 나눈다.
- Models : 64k iterations 에서 학습 종료 
- Dataset : 45k/5k train/val split
- Weight decay : 0.0001
- Momentum : 0.9
- Dropout : ❌(batch norm을 사용했기 때문)

