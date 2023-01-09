# Deep_Learning_Project

## 1. Pneumonia Detection using CNN(92.6% Accuracy)

### 데이터 시각화

- 데이터 개수 (train, test, val) 출력
- countplot을 이용한 개수 시각화 (test, val)
- test, val 이미지 시각화

---

### 조정 전 후 예측 결과 비교



---

### 하이퍼 파라미터 조정

#### 초기 설정

![image](https://user-images.githubusercontent.com/84713532/211253337-4ade8e88-33ba-4bab-a873-3ac1cc84c213.png)

결과: 90%

![image](https://user-images.githubusercontent.com/84713532/211253614-fb31b853-c767-42ae-b3a9-a8c93df15dc5.png)

---

#### rotation_range = 30 -> 60, vertical_flip = False -> True

![image](https://user-images.githubusercontent.com/84713532/211253925-3572aa8f-59b4-4010-9454-4ad7df609e3c.png)

결과: 90%

![image](https://user-images.githubusercontent.com/84713532/211254939-af719fe9-aae1-4124-84ab-25a0fc05e423.png)

---

#### vertical_flip = False -> True

![image](https://user-images.githubusercontent.com/84713532/211255063-c8469386-5858-46a2-ae85-a5e6cf04fbe5.png)

결과: 91% 

![image](https://user-images.githubusercontent.com/84713532/211255508-2ec3dff2-79cf-41f9-abb2-b282459a100f.png)

horizental flip 뿐만 아니라 vertical flip까지 True로 바꿔주니, 예측 확률이 향상되었다.

---

#### zoom_range = 0.2 -> 0.1, width_shift_range = 0.1 -> 0.2, height_shift_range = 0.1 -> 0.2

![image](https://user-images.githubusercontent.com/84713532/211255827-f5e1899b-1040-415f-b8f7-e9fccae418d4.png)

결과: 86%

![image](https://user-images.githubusercontent.com/84713532/211256502-ac228380-d435-4ccf-a91a-cf1e1c6073d3.png)

#### vertical_flip = False -> True, model optimizer = rmsprop -> Adam

![image](https://user-images.githubusercontent.com/84713532/211261808-cdebdede-87c3-43ea-a19c-f29052f0ac0b.png)

결과: 90%

![image](https://user-images.githubusercontent.com/84713532/211262048-32d72c85-eaa4-4de2-be4a-c533f74e608c.png)


**vertical flip=True로 설정해주니 precision과 recall의 개수가 비슷한 것을 확인**

---
        

### 여러 코드 해보기
