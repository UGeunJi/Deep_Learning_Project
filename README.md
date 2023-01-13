# :hospital: Pneumonia Detection with X-Ray Images :skull:

[캐글 X-Ray 폐렴 검출](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

![image](https://user-images.githubusercontent.com/84713532/212211360-e0ac16fe-7681-4f08-9b03-ec05ae6265bc.png)

## :clipboard: Mini Project (2023/01/09 ~ 2023/01/13) :date:

> :family: 팀명: 폐뮤니아
- [이재영](https://github.com/JAYJAY1005)
- [지우근](https://github.com/UGeunJi)
- [주한솔](https://github.com/zzoall)

---

## :scroll: 프로젝트에 대한 전반적인 설명

### 주제 : 딥러닝 예측 모델 성능 올리기

#### 1. 데이터 준비 과정 

```
(0) 시각화 (데이터 증강 전과 후)
(1) 훈련/검증/테스트 데이터 분리
(2) 데이터셋 클래스 정의(자체 제공, 나만의 데이터셋)
(3) 이미지 변환기(torchvision, albumentation, 나만의 전처리기)
(4) 데이터셋 생성/데이터로더 생성
```

#### 2. 모델 생성

```
(1) "나만의 CNN 모델" 만들기 or "이미 학습된 모델" 활용 가능
(2) 손실함수, 옵티마이저, 학습률, 학습 스케쥴러 설정
```

#### 3. 모델 훈련 및 성능 검증

```
(1) 경진대회 아닌 경우 : 평가 (정답이 있음)
(2) 경진대회인 경우 : 예측 및 제출(캐글에서 평가받을 수 있음)
```

---

# :computer: 실행 코드
