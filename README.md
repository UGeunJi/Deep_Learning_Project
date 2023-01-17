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

## 시각화

### 필요한 라이브러리 import

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch # 파이토치 
import random
import os

# 시드값 고정
seed = 50
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
```

```
Using TensorFlow backend.
```

## 데이터 적재

```python
train = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/train')
test = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
val = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/val')
```

```
OpenCV(4.2.0) /io/opencv/modules/imgproc/src/resize.cpp:4045: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
OpenCV(4.2.0) /io/opencv/modules/imgproc/src/resize.cpp:4045: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
OpenCV(4.2.0) /io/opencv/modules/imgproc/src/resize.cpp:4045: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
OpenCV(4.2.0) /io/opencv/modules/imgproc/src/resize.cpp:4045: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
```

```python
print('train:', len(train), '/', 'test:', len(test), '/', 'val:', len(val), '/', 'sum:', len(train) + len(test) + len(val))
```

```
train: 5216 / test: 624 / val: 16 / sum: 5856
```

```python
l = []
for i in train:
    if(i[1] == 0):
        l.append("Pneumonia")
    else:
        l.append("Normal")
sns.set_style('darkgrid')
sns.countplot(l)        
```

![image](https://user-images.githubusercontent.com/84713532/212804217-3ef5dc5b-7984-48c0-ac39-e3e9b6e2b577.png)

```python
print(l.count("Pneumonia"), l.count("Normal"))
```

```
3875 1341
```

```python
ll = []
for i in test:
    if(i[1] == 0):
        ll.append("Pneumonia")
    else:
        ll.append("Normal")
sns.set_style('darkgrid')
sns.countplot(ll)   
```

![image](https://user-images.githubusercontent.com/84713532/212804187-8b061159-b607-4bfd-a939-cc6ac648730c.png)


```python
print(ll.count("Pneumonia"), ll.count("Normal"))
```

```
390 234
```

```python
lll = []
for i in val:
    if(i[1] == 0):
        lll.append("Pneumonia")
    else:
        lll.append("Normal")
sns.set_style('darkgrid')
sns.countplot(lll) 
```

![image](https://user-images.githubusercontent.com/84713532/212804238-69409637-c3cc-4c7e-8ef4-0b452ed2c01d.png)

```python
print(lll.count("Pneumonia"), lll.count("Normal"))
```

```
8 8
```

```python
plt.figure(figsize = (5,5))
plt.imshow(train[500][0], cmap='gray')
plt.title(labels[train[500][1]])

plt.figure(figsize = (5,5))
plt.imshow(train[-2][0], cmap='gray')
plt.title(labels[train[-2][1]])
```

```
Text(0.5, 1.0, 'NORMAL')
```

![image](https://user-images.githubusercontent.com/84713532/212811705-0352fa50-1abe-4965-9802-1ce0351ff829.png)

```python
plt.figure(figsize = (5,5))
plt.imshow(test[0][0], cmap='gray')
plt.title(labels[val[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(test[-2][0], cmap='gray')
plt.title(labels[train[-2][1]])
```

```
Text(0.5, 1.0, 'NORMAL')
```

![image](https://user-images.githubusercontent.com/84713532/212811734-9cb4014a-65ff-4dff-b037-3ace882bf48c.png)

```python
plt.figure(figsize = (5,5))
plt.imshow(val[0][0], cmap='gray')
plt.title(labels[val[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(val[-2][0], cmap='gray')
plt.title(labels[train[-2][1]])
```

```
Text(0.5, 1.0, 'NORMAL')
```

![image](https://user-images.githubusercontent.com/84713532/212811770-19b6514a-997b-4610-a016-8b14860543e3.png)


## 데이터 전처리

```python
x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)
```

```python
# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255
```

```python
# resize data for deep learning 
x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)
```

```python
# With data augmentation to prevent overfitting and handling the imbalance in dataset

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=True)  # randomly flip images


datagen.fit(x_train)
```

## 모델 

```python
model = Sequential()
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1 , activation = 'sigmoid'))
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()
```

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 150, 150, 32)      320       
_________________________________________________________________
batch_normalization_1 (Batch (None, 150, 150, 32)      128       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 75, 75, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 75, 75, 64)        18496     
_________________________________________________________________
dropout_1 (Dropout)          (None, 75, 75, 64)        0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 75, 75, 64)        256       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 38, 38, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 38, 38, 64)        36928     
_________________________________________________________________
batch_normalization_3 (Batch (None, 38, 38, 64)        256       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 19, 19, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 19, 19, 128)       73856     
_________________________________________________________________
dropout_2 (Dropout)          (None, 19, 19, 128)       0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 19, 19, 128)       512       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 10, 10, 128)       0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 10, 10, 256)       295168    
_________________________________________________________________
dropout_3 (Dropout)          (None, 10, 10, 256)       0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 10, 10, 256)       1024      
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 5, 5, 256)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6400)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               819328    
_________________________________________________________________
dropout_4 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 129       
=================================================================
Total params: 1,246,401
Trainable params: 1,245,313
Non-trainable params: 1,088
_________________________________________________________________
```

```python
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 4, verbose=1, factor=0.2, min_lr=0.00000001)
```

```python
history = model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs = 35 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])
```

```
Epoch 1/35
163/163 [==============================] - 14s 83ms/step - loss: 0.5267 - accuracy: 0.8355 - val_loss: 29.7486 - val_accuracy: 0.5000
Epoch 2/35
163/163 [==============================] - 10s 62ms/step - loss: 0.2799 - accuracy: 0.8963 - val_loss: 37.5390 - val_accuracy: 0.5000
Epoch 3/35
163/163 [==============================] - 10s 64ms/step - loss: 0.2437 - accuracy: 0.9149 - val_loss: 15.4706 - val_accuracy: 0.5000
Epoch 4/35
163/163 [==============================] - 11s 67ms/step - loss: 0.2128 - accuracy: 0.9243 - val_loss: 0.4612 - val_accuracy: 0.7500
Epoch 5/35
163/163 [==============================] - 10s 63ms/step - loss: 0.2056 - accuracy: 0.9281 - val_loss: 1.4017 - val_accuracy: 0.5000
Epoch 6/35
163/163 [==============================] - 10s 64ms/step - loss: 0.1809 - accuracy: 0.9369 - val_loss: 1.0402 - val_accuracy: 0.6875
Epoch 7/35
163/163 [==============================] - 11s 69ms/step - loss: 0.1659 - accuracy: 0.9479 - val_loss: 0.9123 - val_accuracy: 0.6875
Epoch 8/35
163/163 [==============================] - 10s 64ms/step - loss: 0.1605 - accuracy: 0.9494 - val_loss: 1.5733 - val_accuracy: 0.4375

Epoch 00008: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
Epoch 9/35
163/163 [==============================] - 11s 65ms/step - loss: 0.1194 - accuracy: 0.9594 - val_loss: 16.6094 - val_accuracy: 0.5000
Epoch 10/35
163/163 [==============================] - 11s 68ms/step - loss: 0.1063 - accuracy: 0.9632 - val_loss: 3.3700 - val_accuracy: 0.5625
Epoch 11/35
163/163 [==============================] - 11s 66ms/step - loss: 0.1089 - accuracy: 0.9664 - val_loss: 0.9141 - val_accuracy: 0.5625
Epoch 12/35
163/163 [==============================] - 11s 65ms/step - loss: 0.1040 - accuracy: 0.9640 - val_loss: 1.0078 - val_accuracy: 0.6250

Epoch 00012: ReduceLROnPlateau reducing learning rate to 4.0000001899898055e-05.
Epoch 13/35
163/163 [==============================] - 11s 67ms/step - loss: 0.0912 - accuracy: 0.9668 - val_loss: 0.3887 - val_accuracy: 0.8125
Epoch 14/35
163/163 [==============================] - 10s 64ms/step - loss: 0.0860 - accuracy: 0.9707 - val_loss: 5.4680 - val_accuracy: 0.5000
Epoch 15/35
163/163 [==============================] - 10s 64ms/step - loss: 0.0989 - accuracy: 0.9707 - val_loss: 1.9474 - val_accuracy: 0.5625
Epoch 16/35
163/163 [==============================] - 11s 68ms/step - loss: 0.1002 - accuracy: 0.9699 - val_loss: 5.9309 - val_accuracy: 0.5000
Epoch 17/35
163/163 [==============================] - 11s 64ms/step - loss: 0.0774 - accuracy: 0.9712 - val_loss: 7.4238 - val_accuracy: 0.6250

Epoch 00017: ReduceLROnPlateau reducing learning rate to 8.000000525498762e-06.
Epoch 18/35
163/163 [==============================] - 10s 64ms/step - loss: 0.0902 - accuracy: 0.9707 - val_loss: 0.4661 - val_accuracy: 0.8125
Epoch 19/35
163/163 [==============================] - 11s 66ms/step - loss: 0.0836 - accuracy: 0.9722 - val_loss: 0.5257 - val_accuracy: 0.5625
Epoch 20/35
163/163 [==============================] - 10s 63ms/step - loss: 0.0803 - accuracy: 0.9720 - val_loss: 0.5235 - val_accuracy: 0.5625
Epoch 21/35
163/163 [==============================] - 10s 63ms/step - loss: 0.0886 - accuracy: 0.9659 - val_loss: 0.8678 - val_accuracy: 0.6875

Epoch 00021: ReduceLROnPlateau reducing learning rate to 1.6000001778593287e-06.
Epoch 22/35
163/163 [==============================] - 11s 67ms/step - loss: 0.1011 - accuracy: 0.9657 - val_loss: 1.3487 - val_accuracy: 0.6250
Epoch 23/35
163/163 [==============================] - 10s 64ms/step - loss: 0.0923 - accuracy: 0.9703 - val_loss: 1.3251 - val_accuracy: 0.6250
Epoch 24/35
163/163 [==============================] - 10s 62ms/step - loss: 0.0823 - accuracy: 0.9705 - val_loss: 0.8454 - val_accuracy: 0.6250
Epoch 25/35
163/163 [==============================] - 11s 67ms/step - loss: 0.0919 - accuracy: 0.9674 - val_loss: 0.7879 - val_accuracy: 0.6250

Epoch 00025: ReduceLROnPlateau reducing learning rate to 3.200000264769187e-07.
Epoch 26/35
163/163 [==============================] - 11s 65ms/step - loss: 0.0833 - accuracy: 0.9720 - val_loss: 0.9681 - val_accuracy: 0.5625
Epoch 27/35
163/163 [==============================] - 10s 63ms/step - loss: 0.0936 - accuracy: 0.9707 - val_loss: 1.1970 - val_accuracy: 0.6250
Epoch 28/35
163/163 [==============================] - 11s 66ms/step - loss: 0.0878 - accuracy: 0.9709 - val_loss: 1.1222 - val_accuracy: 0.6875
Epoch 29/35
163/163 [==============================] - 10s 63ms/step - loss: 0.0870 - accuracy: 0.9720 - val_loss: 0.9915 - val_accuracy: 0.5000

Epoch 00029: ReduceLROnPlateau reducing learning rate to 6.400000529538374e-08.
Epoch 30/35
163/163 [==============================] - 10s 64ms/step - loss: 0.0846 - accuracy: 0.9711 - val_loss: 0.8008 - val_accuracy: 0.6250
Epoch 31/35
163/163 [==============================] - 11s 65ms/step - loss: 0.0822 - accuracy: 0.9741 - val_loss: 0.9858 - val_accuracy: 0.6250
Epoch 32/35
163/163 [==============================] - 10s 64ms/step - loss: 0.0833 - accuracy: 0.9720 - val_loss: 0.8316 - val_accuracy: 0.6250
Epoch 33/35
163/163 [==============================] - 11s 65ms/step - loss: 0.0873 - accuracy: 0.9732 - val_loss: 0.9309 - val_accuracy: 0.6250

Epoch 00033: ReduceLROnPlateau reducing learning rate to 1.2800001059076749e-08.
Epoch 34/35
163/163 [==============================] - 11s 65ms/step - loss: 0.0880 - accuracy: 0.9693 - val_loss: 0.7870 - val_accuracy: 0.5625
Epoch 35/35
163/163 [==============================] - 10s 64ms/step - loss: 0.0955 - accuracy: 0.9695 - val_loss: 1.7780 - val_accuracy: 0.6250
```

```python
print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")
```
### 결과 출력

```
624/624 [==============================] - 0s 492us/step
Loss of the model is -  0.23225767528399444
624/624 [==============================] - 0s 468us/step
Accuracy of the model is -  92.78292170143127 %
```

# Analysis after Model Training

```python
epochs = [i for i in range(35)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()
```

![image](https://user-images.githubusercontent.com/84713532/212812089-56237b4c-c103-4dce-a743-62a133e4506b.png)

```python
predictions = model.predict_classes(x_test)
predictions = predictions.reshape(1,-1)[0]
predictions[:15]
```

```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=int32)
```

```python
print(classification_report(y_test, predictions, target_names = ['Pneumonia (Class 0)','Normal (Class 1)']))
```

```
                     precision    recall  f1-score   support

Pneumonia (Class 0)       0.92      0.95      0.94       390
   Normal (Class 1)       0.91      0.86      0.89       234

           accuracy                           0.92       624
          macro avg       0.92      0.91      0.91       624
       weighted avg       0.92      0.92      0.92       624
```

```python
cm = confusion_matrix(y_test,predictions)
cm
```

```
array([[371,  19],
       [ 32, 202]])
```

```python
cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
```

```python
plt.figure(figsize = (10,10))
sns.heatmap(cm, cmap= "Greens", linecolor = 'White' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)
```

```
<matplotlib.axes._subplots.AxesSubplot at 0x7f60185578d0>
```

![image](https://user-images.githubusercontent.com/84713532/212812130-d2212667-a7bb-4017-a118-e2a6040f0e6c.png)


```python
correct = np.nonzero(predictions == y_test)[0]
incorrect = np.nonzero(predictions != y_test)[0]
```

**Some of the Correctly Predicted Classes**

```python
i = 0
for c in correct[:6]:
    plt.subplot(3,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[c].reshape(150,150), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y_test[c]))
    plt.tight_layout()
    i += 1
```

![image](https://user-images.githubusercontent.com/84713532/212812180-f67cbc05-140f-4264-8d9c-0c20a6b7d74e.png)


**Some of the Incorrectly Predicted Classes**

```python
i = 0
for c in incorrect[2:10]:
    plt.subplot(4,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[c].reshape(150,150), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y_test[c]))
    plt.tight_layout()
    i += 1
```

![image](https://user-images.githubusercontent.com/84713532/212812225-37381fef-b4b9-454b-882a-ba1123999553.png)

--- 

## 파라미터 조정

- Augmentation: vertical flip=False -> True
- ReduceLROnPlateau: patience=2 -> 4, factor=0.3 -> 0.2, min_lr=0.000001 -> 0.00000001
- fit: epoch=12 -> 35
- 성능 변화: 92.6 -> 92.78

#### 파라미터 조정 실패

![image](https://user-images.githubusercontent.com/84713532/212813061-01ba6b47-ac9d-4aa5-b546-78c2a34d3307.png)

## 성능 낮은 모델들(VGGNet, GoogleNet, Resnet)

**VGGNet**

```py
# import libraries
import tensorflow as tf
import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

```py
# VGG Structure
IMAGE_SIZE = 150

def build_vgg16():
  tf.keras.backend.clear_session()
  input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    
  x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block1_conv1")(input_tensor)
  x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block1_conv2")(x)  
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool")(x) 

  x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block2_conv1")(x)
  x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block2_conv2")(x)  
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool")(x) 
    
  x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block3_conv1")(x)
  x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block3_conv2")(x)  
  x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block3_conv3")(x)  
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool")(x) 

  x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block4_conv1")(x)
  x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block4_conv2")(x)  
  x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block4_conv3")(x)  
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool")(x) 
    
  x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block5_conv1")(x)
  x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block5_conv2")(x)  
  x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="same", activation="relu", name="block5_conv3")(x)  
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block5_pool")(x)   # 7*7*512
    

  x = GlobalAveragePooling2D()(x) # 512
  x = Dense(50, activation="relu")(x)
  output = Dense(units=1, activation='sigmoid')(x)


  model = Model(inputs=input_tensor, outputs=output)
  model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'] )
  return model

model = build_vgg16()
model.summary()
```

```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 150, 150, 1)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 150, 150, 64)      640       
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
_________________________________________________________________
global_average_pooling2d (Gl (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 50)                25650     
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 51        
=================================================================
Total params: 14,739,237
Trainable params: 14,739,237
Non-trainable params: 0
_________________________________________________________________
```

**Result**
![image](https://user-images.githubusercontent.com/84713532/212813656-0372f027-62a2-42ea-92cd-53e463aed7f5.png)

---

**GoogleNet**

```py
# import libraries
import tensorflow as tf
import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

```py
# GoogleNet structure
IMAGE_SIZE = 150

def build_GoogLeNet():
    tf.keras.backend.clear_session()
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    
    x = Conv2D(filters=64, kernel_size=(7, 7), padding="same", strides=(2, 2), activation="relu")(input_tensor)
    x = MaxPooling2D(pool_size=(3, 3), padding="same", strides=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(1, 1), padding="same", strides=(1, 1), activation="relu")(x)
    x = Conv2D(filters=192, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), padding="same", strides=(2, 2))(x)
    
    # 인셉션 모듈 3a
    x = inception_module(x, 64, 96, 128, 16, 32, 32, name="inception_3a")
    # 인셉션 모듈 3b
    x = inception_module(x, 128, 128, 192, 32, 96, 64, name="inception_3b")

    x = MaxPooling2D(pool_size=(3, 3), padding="same", strides=(2, 2))(x)

    # 인셉션 모듈 4a
    x = inception_module(x, 192, 96, 208, 16, 48, 64, "inception_4a")
    # 인셉션 모듈 4b
    x = inception_module(x, 160, 112, 224, 24, 64, 64, name="inception_4b")
    # 인셉션 모듈 4c
    x = inception_module(x, 128, 128, 256, 24, 64, 64, name="inception_4c")
    # 인셉션 모듈 4d
    x = inception_module(x, 112, 144, 288, 32, 64, 64, name="inception_4d")
    # 인셉션 모듈 4e
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name="inception_4e")

    x = MaxPooling2D(pool_size=(3, 3), padding="same", strides=(2, 2))(x)

    # 인셉션 모듈 5a
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name="inception_5a")
    # 인셉션 모듈 5b
    x = inception_module(x, 384, 192, 384, 48, 128, 128, name="inception_5b")

    x = GlobalAveragePooling2D()(x) 
    x = Dropout(0.4)(x)    
    output = Dense(units=1, activation='sigmoid')(x)


    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'] )
    return model

model = build_GoogLeNet()
model.summary()
```

```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 150, 150, 1) 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 75, 75, 64)   3200        input_1[0][0]                    
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 38, 38, 64)   0           conv2d[0][0]                     
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 38, 38, 64)   4160        max_pooling2d[0][0]              
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 38, 38, 192)  110784      conv2d_1[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 19, 19, 192)  0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 19, 19, 96)   18528       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 19, 19, 16)   3088        max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 19, 19, 192)  0           max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 19, 19, 64)   12352       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 19, 19, 128)  110720      conv2d_4[0][0]                   
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 19, 19, 32)   12832       conv2d_6[0][0]                   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 19, 19, 32)   6176        max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
inception_3a (Concatenate)      (None, 19, 19, 256)  0           conv2d_3[0][0]                   
                                                                 conv2d_5[0][0]                   
                                                                 conv2d_7[0][0]                   
                                                                 conv2d_8[0][0]                   
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 19, 19, 128)  32896       inception_3a[0][0]               
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 19, 19, 32)   8224        inception_3a[0][0]               
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 19, 19, 256)  0           inception_3a[0][0]               
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 19, 19, 128)  32896       inception_3a[0][0]               
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 19, 19, 192)  221376      conv2d_10[0][0]                  
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 19, 19, 96)   76896       conv2d_12[0][0]                  
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 19, 19, 64)   16448       max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
inception_3b (Concatenate)      (None, 19, 19, 480)  0           conv2d_9[0][0]                   
                                                                 conv2d_11[0][0]                  
                                                                 conv2d_13[0][0]                  
                                                                 conv2d_14[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 10, 10, 480)  0           inception_3b[0][0]               
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 10, 10, 96)   46176       max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 10, 10, 16)   7696        max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 10, 10, 480)  0           max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 10, 10, 192)  92352       max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 10, 10, 208)  179920      conv2d_16[0][0]                  
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 10, 10, 48)   19248       conv2d_18[0][0]                  
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 10, 10, 64)   30784       max_pooling2d_5[0][0]            
__________________________________________________________________________________________________
inception_4a (Concatenate)      (None, 10, 10, 512)  0           conv2d_15[0][0]                  
                                                                 conv2d_17[0][0]                  
                                                                 conv2d_19[0][0]                  
                                                                 conv2d_20[0][0]                  
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 10, 10, 112)  57456       inception_4a[0][0]               
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 10, 10, 24)   12312       inception_4a[0][0]               
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 10, 10, 512)  0           inception_4a[0][0]               
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 10, 10, 160)  82080       inception_4a[0][0]               
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 10, 10, 224)  226016      conv2d_22[0][0]                  
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 10, 10, 64)   38464       conv2d_24[0][0]                  
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 10, 10, 64)   32832       max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
inception_4b (Concatenate)      (None, 10, 10, 512)  0           conv2d_21[0][0]                  
                                                                 conv2d_23[0][0]                  
                                                                 conv2d_25[0][0]                  
                                                                 conv2d_26[0][0]                  
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 10, 10, 128)  65664       inception_4b[0][0]               
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 10, 10, 24)   12312       inception_4b[0][0]               
__________________________________________________________________________________________________
max_pooling2d_7 (MaxPooling2D)  (None, 10, 10, 512)  0           inception_4b[0][0]               
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 10, 10, 128)  65664       inception_4b[0][0]               
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 10, 10, 256)  295168      conv2d_28[0][0]                  
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 10, 10, 64)   38464       conv2d_30[0][0]                  
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 10, 10, 64)   32832       max_pooling2d_7[0][0]            
__________________________________________________________________________________________________
inception_4c (Concatenate)      (None, 10, 10, 512)  0           conv2d_27[0][0]                  
                                                                 conv2d_29[0][0]                  
                                                                 conv2d_31[0][0]                  
                                                                 conv2d_32[0][0]                  
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 10, 10, 144)  73872       inception_4c[0][0]               
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 10, 10, 32)   16416       inception_4c[0][0]               
__________________________________________________________________________________________________
max_pooling2d_8 (MaxPooling2D)  (None, 10, 10, 512)  0           inception_4c[0][0]               
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 10, 10, 112)  57456       inception_4c[0][0]               
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 10, 10, 288)  373536      conv2d_34[0][0]                  
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 10, 10, 64)   51264       conv2d_36[0][0]                  
__________________________________________________________________________________________________
conv2d_38 (Conv2D)              (None, 10, 10, 64)   32832       max_pooling2d_8[0][0]            
__________________________________________________________________________________________________
inception_4d (Concatenate)      (None, 10, 10, 528)  0           conv2d_33[0][0]                  
                                                                 conv2d_35[0][0]                  
                                                                 conv2d_37[0][0]                  
                                                                 conv2d_38[0][0]                  
__________________________________________________________________________________________________
conv2d_40 (Conv2D)              (None, 10, 10, 160)  84640       inception_4d[0][0]               
__________________________________________________________________________________________________
conv2d_42 (Conv2D)              (None, 10, 10, 32)   16928       inception_4d[0][0]               
__________________________________________________________________________________________________
max_pooling2d_9 (MaxPooling2D)  (None, 10, 10, 528)  0           inception_4d[0][0]               
__________________________________________________________________________________________________
conv2d_39 (Conv2D)              (None, 10, 10, 256)  135424      inception_4d[0][0]               
__________________________________________________________________________________________________
conv2d_41 (Conv2D)              (None, 10, 10, 320)  461120      conv2d_40[0][0]                  
__________________________________________________________________________________________________
conv2d_43 (Conv2D)              (None, 10, 10, 128)  102528      conv2d_42[0][0]                  
__________________________________________________________________________________________________
conv2d_44 (Conv2D)              (None, 10, 10, 128)  67712       max_pooling2d_9[0][0]            
__________________________________________________________________________________________________
inception_4e (Concatenate)      (None, 10, 10, 832)  0           conv2d_39[0][0]                  
                                                                 conv2d_41[0][0]                  
                                                                 conv2d_43[0][0]                  
                                                                 conv2d_44[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_10 (MaxPooling2D) (None, 5, 5, 832)    0           inception_4e[0][0]               
__________________________________________________________________________________________________
conv2d_46 (Conv2D)              (None, 5, 5, 160)    133280      max_pooling2d_10[0][0]           
__________________________________________________________________________________________________
conv2d_48 (Conv2D)              (None, 5, 5, 32)     26656       max_pooling2d_10[0][0]           
__________________________________________________________________________________________________
max_pooling2d_11 (MaxPooling2D) (None, 5, 5, 832)    0           max_pooling2d_10[0][0]           
__________________________________________________________________________________________________
conv2d_45 (Conv2D)              (None, 5, 5, 256)    213248      max_pooling2d_10[0][0]           
__________________________________________________________________________________________________
conv2d_47 (Conv2D)              (None, 5, 5, 320)    461120      conv2d_46[0][0]                  
__________________________________________________________________________________________________
conv2d_49 (Conv2D)              (None, 5, 5, 128)    102528      conv2d_48[0][0]                  
__________________________________________________________________________________________________
conv2d_50 (Conv2D)              (None, 5, 5, 128)    106624      max_pooling2d_11[0][0]           
__________________________________________________________________________________________________
inception_5a (Concatenate)      (None, 5, 5, 832)    0           conv2d_45[0][0]                  
                                                                 conv2d_47[0][0]                  
                                                                 conv2d_49[0][0]                  
                                                                 conv2d_50[0][0]                  
__________________________________________________________________________________________________
conv2d_52 (Conv2D)              (None, 5, 5, 192)    159936      inception_5a[0][0]               
__________________________________________________________________________________________________
conv2d_54 (Conv2D)              (None, 5, 5, 48)     39984       inception_5a[0][0]               
__________________________________________________________________________________________________
max_pooling2d_12 (MaxPooling2D) (None, 5, 5, 832)    0           inception_5a[0][0]               
__________________________________________________________________________________________________
conv2d_51 (Conv2D)              (None, 5, 5, 384)    319872      inception_5a[0][0]               
__________________________________________________________________________________________________
conv2d_53 (Conv2D)              (None, 5, 5, 384)    663936      conv2d_52[0][0]                  
__________________________________________________________________________________________________
conv2d_55 (Conv2D)              (None, 5, 5, 128)    153728      conv2d_54[0][0]                  
__________________________________________________________________________________________________
conv2d_56 (Conv2D)              (None, 5, 5, 128)    106624      max_pooling2d_12[0][0]           
__________________________________________________________________________________________________
inception_5b (Concatenate)      (None, 5, 5, 1024)   0           conv2d_51[0][0]                  
                                                                 conv2d_53[0][0]                  
                                                                 conv2d_55[0][0]                  
                                                                 conv2d_56[0][0]                  
__________________________________________________________________________________________________
global_average_pooling2d (Globa (None, 1024)         0           inception_5b[0][0]               
__________________________________________________________________________________________________
dropout (Dropout)               (None, 1024)         0           global_average_pooling2d[0][0]   
__________________________________________________________________________________________________
dense (Dense)                   (None, 1)            1025        dropout[0][0]                    
==================================================================================================
Total params: 5,968,305
Trainable params: 5,968,305
Non-trainable params: 0
__________________________________________________________________________________________________
```

**Result**
![image](https://user-images.githubusercontent.com/84713532/212813990-6cab3a27-a118-4ac1-b532-b5daefbd7eb5.png)

---

**Resnet**

```py
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Activation, MaxPooling2D, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Add, ZeroPadding2D
from tensorflow.keras.callbacks import TensorBoard
```

```py
# model structure
def identity_block(input_tensor, filters):
    filter1, filter2, filter3 = filters
    
    x = Conv2D(filters=filter1, kernel_size=(1, 1), padding="same", kernel_initializer="he_normal")(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters=filter2, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters=filter3, kernel_size=(1, 1), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization(axis=3)(x)

    x = Add()([input_tensor, x])
    output = Activation("relu")(x)
    
    return output
```
```py
def convolutional_block(input_tensor, filters, strides=(2,2)):
    filter1, filter2, filter3 = filters
    
    x = Conv2D(filters=filter1, kernel_size=(1, 1), padding="same", strides = strides, kernel_initializer="he_normal")(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters=filter2, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters=filter3, kernel_size=(1, 1), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization(axis=3)(x)

    shortcut = Conv2D(filters=filter3, kernel_size=(1, 1), padding="same", strides=strides, kernel_initializer="he_normal")(input_tensor)
    shortcut = BatchNormalization(axis=3)(shortcut)
    
    x = Add()([x, shortcut])
    output = Activation("relu")(x)
    
    return output
```
```py
def build_resnet50():
    tf.keras.backend.clear_session()
    input_tensor = Input(shape=(150, 150, 1))

    x = ZeroPadding2D(padding=(3,3))(input_tensor)
    x = Conv2D(filters=64, kernel_size=(7,7), padding="valid", strides=(2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    x = ZeroPadding2D(padding=(1,1))(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = convolutional_block(x, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    x = convolutional_block(x, [128, 128, 512], strides=(2, 2))
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    x = convolutional_block(x, [256, 256, 1024], strides=(2, 2))
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    x = convolutional_block(x, [512, 512, 2048], strides=(2, 2))
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(200, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_tensor, outputs= output)
    model.compile(optimizer=SGD(0.01, momentum=0.9), loss="binary_crossentropy", metrics=["accuracy"])
    
    return model

model = build_resnet50()
model.summary()
```

```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 150, 150, 1) 0                                            
__________________________________________________________________________________________________
zero_padding2d (ZeroPadding2D)  (None, 156, 156, 1)  0           input_1[0][0]                    
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 75, 75, 64)   3200        zero_padding2d[0][0]             
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 75, 75, 64)   256         conv2d[0][0]                     
__________________________________________________________________________________________________
activation (Activation)         (None, 75, 75, 64)   0           batch_normalization[0][0]        
__________________________________________________________________________________________________
zero_padding2d_1 (ZeroPadding2D (None, 77, 77, 64)   0           activation[0][0]                 
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 38, 38, 64)   0           zero_padding2d_1[0][0]           
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 38, 38, 64)   4160        max_pooling2d[0][0]              
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 38, 38, 64)   256         conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 38, 38, 64)   0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 38, 38, 64)   36928       activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 38, 38, 64)   256         conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 38, 38, 64)   0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 38, 38, 256)  16640       activation_2[0][0]               
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 38, 38, 256)  16640       max_pooling2d[0][0]              
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 38, 38, 256)  1024        conv2d_3[0][0]                   
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 38, 38, 256)  1024        conv2d_4[0][0]                   
__________________________________________________________________________________________________
add (Add)                       (None, 38, 38, 256)  0           batch_normalization_3[0][0]      
                                                                 batch_normalization_4[0][0]      
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 38, 38, 256)  0           add[0][0]                        
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 38, 38, 64)   16448       activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 38, 38, 64)   256         conv2d_5[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 38, 38, 64)   0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 38, 38, 64)   36928       activation_4[0][0]               
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 38, 38, 64)   256         conv2d_6[0][0]                   
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 38, 38, 64)   0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 38, 38, 256)  16640       activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 38, 38, 256)  1024        conv2d_7[0][0]                   
__________________________________________________________________________________________________
add_1 (Add)                     (None, 38, 38, 256)  0           activation_3[0][0]               
                                                                 batch_normalization_7[0][0]      
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 38, 38, 256)  0           add_1[0][0]                      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 38, 38, 64)   16448       activation_6[0][0]               
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 38, 38, 64)   256         conv2d_8[0][0]                   
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 38, 38, 64)   0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 38, 38, 64)   36928       activation_7[0][0]               
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 38, 38, 64)   256         conv2d_9[0][0]                   
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 38, 38, 64)   0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 38, 38, 256)  16640       activation_8[0][0]               
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 38, 38, 256)  1024        conv2d_10[0][0]                  
__________________________________________________________________________________________________
add_2 (Add)                     (None, 38, 38, 256)  0           activation_6[0][0]               
                                                                 batch_normalization_10[0][0]     
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 38, 38, 256)  0           add_2[0][0]                      
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 19, 19, 128)  32896       activation_9[0][0]               
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 19, 19, 128)  512         conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 19, 19, 128)  0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 19, 19, 128)  147584      activation_10[0][0]              
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 19, 19, 128)  512         conv2d_12[0][0]                  
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 19, 19, 128)  0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 19, 19, 512)  66048       activation_11[0][0]              
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 19, 19, 512)  131584      activation_9[0][0]               
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 19, 19, 512)  2048        conv2d_13[0][0]                  
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 19, 19, 512)  2048        conv2d_14[0][0]                  
__________________________________________________________________________________________________
add_3 (Add)                     (None, 19, 19, 512)  0           batch_normalization_13[0][0]     
                                                                 batch_normalization_14[0][0]     
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 19, 19, 512)  0           add_3[0][0]                      
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 19, 19, 128)  65664       activation_12[0][0]              
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 19, 19, 128)  512         conv2d_15[0][0]                  
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 19, 19, 128)  0           batch_normalization_15[0][0]     
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 19, 19, 128)  147584      activation_13[0][0]              
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 19, 19, 128)  512         conv2d_16[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 19, 19, 128)  0           batch_normalization_16[0][0]     
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 19, 19, 512)  66048       activation_14[0][0]              
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 19, 19, 512)  2048        conv2d_17[0][0]                  
__________________________________________________________________________________________________
add_4 (Add)                     (None, 19, 19, 512)  0           activation_12[0][0]              
                                                                 batch_normalization_17[0][0]     
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 19, 19, 512)  0           add_4[0][0]                      
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 19, 19, 128)  65664       activation_15[0][0]              
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 19, 19, 128)  512         conv2d_18[0][0]                  
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 19, 19, 128)  0           batch_normalization_18[0][0]     
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 19, 19, 128)  147584      activation_16[0][0]              
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 19, 19, 128)  512         conv2d_19[0][0]                  
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 19, 19, 128)  0           batch_normalization_19[0][0]     
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 19, 19, 512)  66048       activation_17[0][0]              
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 19, 19, 512)  2048        conv2d_20[0][0]                  
__________________________________________________________________________________________________
add_5 (Add)                     (None, 19, 19, 512)  0           activation_15[0][0]              
                                                                 batch_normalization_20[0][0]     
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 19, 19, 512)  0           add_5[0][0]                      
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 19, 19, 128)  65664       activation_18[0][0]              
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 19, 19, 128)  512         conv2d_21[0][0]                  
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 19, 19, 128)  0           batch_normalization_21[0][0]     
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 19, 19, 128)  147584      activation_19[0][0]              
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 19, 19, 128)  512         conv2d_22[0][0]                  
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 19, 19, 128)  0           batch_normalization_22[0][0]     
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 19, 19, 512)  66048       activation_20[0][0]              
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 19, 19, 512)  2048        conv2d_23[0][0]                  
__________________________________________________________________________________________________
add_6 (Add)                     (None, 19, 19, 512)  0           activation_18[0][0]              
                                                                 batch_normalization_23[0][0]     
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 19, 19, 512)  0           add_6[0][0]                      
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 10, 10, 256)  131328      activation_21[0][0]              
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 10, 10, 256)  1024        conv2d_24[0][0]                  
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 10, 10, 256)  0           batch_normalization_24[0][0]     
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 10, 10, 256)  590080      activation_22[0][0]              
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 10, 10, 256)  1024        conv2d_25[0][0]                  
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 10, 10, 256)  0           batch_normalization_25[0][0]     
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 10, 10, 1024) 263168      activation_23[0][0]              
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 10, 10, 1024) 525312      activation_21[0][0]              
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 10, 10, 1024) 4096        conv2d_26[0][0]                  
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 10, 10, 1024) 4096        conv2d_27[0][0]                  
__________________________________________________________________________________________________
add_7 (Add)                     (None, 10, 10, 1024) 0           batch_normalization_26[0][0]     
                                                                 batch_normalization_27[0][0]     
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 10, 10, 1024) 0           add_7[0][0]                      
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 10, 10, 256)  262400      activation_24[0][0]              
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 10, 10, 256)  1024        conv2d_28[0][0]                  
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 10, 10, 256)  0           batch_normalization_28[0][0]     
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 10, 10, 256)  590080      activation_25[0][0]              
__________________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, 10, 10, 256)  1024        conv2d_29[0][0]                  
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 10, 10, 256)  0           batch_normalization_29[0][0]     
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 10, 10, 1024) 263168      activation_26[0][0]              
__________________________________________________________________________________________________
batch_normalization_30 (BatchNo (None, 10, 10, 1024) 4096        conv2d_30[0][0]                  
__________________________________________________________________________________________________
add_8 (Add)                     (None, 10, 10, 1024) 0           activation_24[0][0]              
                                                                 batch_normalization_30[0][0]     
__________________________________________________________________________________________________
activation_27 (Activation)      (None, 10, 10, 1024) 0           add_8[0][0]                      
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 10, 10, 256)  262400      activation_27[0][0]              
__________________________________________________________________________________________________
batch_normalization_31 (BatchNo (None, 10, 10, 256)  1024        conv2d_31[0][0]                  
__________________________________________________________________________________________________
activation_28 (Activation)      (None, 10, 10, 256)  0           batch_normalization_31[0][0]     
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 10, 10, 256)  590080      activation_28[0][0]              
__________________________________________________________________________________________________
batch_normalization_32 (BatchNo (None, 10, 10, 256)  1024        conv2d_32[0][0]                  
__________________________________________________________________________________________________
activation_29 (Activation)      (None, 10, 10, 256)  0           batch_normalization_32[0][0]     
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 10, 10, 1024) 263168      activation_29[0][0]              
__________________________________________________________________________________________________
batch_normalization_33 (BatchNo (None, 10, 10, 1024) 4096        conv2d_33[0][0]                  
__________________________________________________________________________________________________
add_9 (Add)                     (None, 10, 10, 1024) 0           activation_27[0][0]              
                                                                 batch_normalization_33[0][0]     
__________________________________________________________________________________________________
activation_30 (Activation)      (None, 10, 10, 1024) 0           add_9[0][0]                      
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 10, 10, 256)  262400      activation_30[0][0]              
__________________________________________________________________________________________________
batch_normalization_34 (BatchNo (None, 10, 10, 256)  1024        conv2d_34[0][0]                  
__________________________________________________________________________________________________
activation_31 (Activation)      (None, 10, 10, 256)  0           batch_normalization_34[0][0]     
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 10, 10, 256)  590080      activation_31[0][0]              
__________________________________________________________________________________________________
batch_normalization_35 (BatchNo (None, 10, 10, 256)  1024        conv2d_35[0][0]                  
__________________________________________________________________________________________________
activation_32 (Activation)      (None, 10, 10, 256)  0           batch_normalization_35[0][0]     
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 10, 10, 1024) 263168      activation_32[0][0]              
__________________________________________________________________________________________________
batch_normalization_36 (BatchNo (None, 10, 10, 1024) 4096        conv2d_36[0][0]                  
__________________________________________________________________________________________________
add_10 (Add)                    (None, 10, 10, 1024) 0           activation_30[0][0]              
                                                                 batch_normalization_36[0][0]     
__________________________________________________________________________________________________
activation_33 (Activation)      (None, 10, 10, 1024) 0           add_10[0][0]                     
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 10, 10, 256)  262400      activation_33[0][0]              
__________________________________________________________________________________________________
batch_normalization_37 (BatchNo (None, 10, 10, 256)  1024        conv2d_37[0][0]                  
__________________________________________________________________________________________________
activation_34 (Activation)      (None, 10, 10, 256)  0           batch_normalization_37[0][0]     
__________________________________________________________________________________________________
conv2d_38 (Conv2D)              (None, 10, 10, 256)  590080      activation_34[0][0]              
__________________________________________________________________________________________________
batch_normalization_38 (BatchNo (None, 10, 10, 256)  1024        conv2d_38[0][0]                  
__________________________________________________________________________________________________
activation_35 (Activation)      (None, 10, 10, 256)  0           batch_normalization_38[0][0]     
__________________________________________________________________________________________________
conv2d_39 (Conv2D)              (None, 10, 10, 1024) 263168      activation_35[0][0]              
__________________________________________________________________________________________________
batch_normalization_39 (BatchNo (None, 10, 10, 1024) 4096        conv2d_39[0][0]                  
__________________________________________________________________________________________________
add_11 (Add)                    (None, 10, 10, 1024) 0           activation_33[0][0]              
                                                                 batch_normalization_39[0][0]     
__________________________________________________________________________________________________
activation_36 (Activation)      (None, 10, 10, 1024) 0           add_11[0][0]                     
__________________________________________________________________________________________________
conv2d_40 (Conv2D)              (None, 10, 10, 256)  262400      activation_36[0][0]              
__________________________________________________________________________________________________
batch_normalization_40 (BatchNo (None, 10, 10, 256)  1024        conv2d_40[0][0]                  
__________________________________________________________________________________________________
activation_37 (Activation)      (None, 10, 10, 256)  0           batch_normalization_40[0][0]     
__________________________________________________________________________________________________
conv2d_41 (Conv2D)              (None, 10, 10, 256)  590080      activation_37[0][0]              
__________________________________________________________________________________________________
batch_normalization_41 (BatchNo (None, 10, 10, 256)  1024        conv2d_41[0][0]                  
__________________________________________________________________________________________________
activation_38 (Activation)      (None, 10, 10, 256)  0           batch_normalization_41[0][0]     
__________________________________________________________________________________________________
conv2d_42 (Conv2D)              (None, 10, 10, 1024) 263168      activation_38[0][0]              
__________________________________________________________________________________________________
batch_normalization_42 (BatchNo (None, 10, 10, 1024) 4096        conv2d_42[0][0]                  
__________________________________________________________________________________________________
add_12 (Add)                    (None, 10, 10, 1024) 0           activation_36[0][0]              
                                                                 batch_normalization_42[0][0]     
__________________________________________________________________________________________________
activation_39 (Activation)      (None, 10, 10, 1024) 0           add_12[0][0]                     
__________________________________________________________________________________________________
conv2d_43 (Conv2D)              (None, 5, 5, 512)    524800      activation_39[0][0]              
__________________________________________________________________________________________________
batch_normalization_43 (BatchNo (None, 5, 5, 512)    2048        conv2d_43[0][0]                  
__________________________________________________________________________________________________
activation_40 (Activation)      (None, 5, 5, 512)    0           batch_normalization_43[0][0]     
__________________________________________________________________________________________________
conv2d_44 (Conv2D)              (None, 5, 5, 512)    2359808     activation_40[0][0]              
__________________________________________________________________________________________________
batch_normalization_44 (BatchNo (None, 5, 5, 512)    2048        conv2d_44[0][0]                  
__________________________________________________________________________________________________
activation_41 (Activation)      (None, 5, 5, 512)    0           batch_normalization_44[0][0]     
__________________________________________________________________________________________________
conv2d_45 (Conv2D)              (None, 5, 5, 2048)   1050624     activation_41[0][0]              
__________________________________________________________________________________________________
conv2d_46 (Conv2D)              (None, 5, 5, 2048)   2099200     activation_39[0][0]              
__________________________________________________________________________________________________
batch_normalization_45 (BatchNo (None, 5, 5, 2048)   8192        conv2d_45[0][0]                  
__________________________________________________________________________________________________
batch_normalization_46 (BatchNo (None, 5, 5, 2048)   8192        conv2d_46[0][0]                  
__________________________________________________________________________________________________
add_13 (Add)                    (None, 5, 5, 2048)   0           batch_normalization_45[0][0]     
                                                                 batch_normalization_46[0][0]     
__________________________________________________________________________________________________
activation_42 (Activation)      (None, 5, 5, 2048)   0           add_13[0][0]                     
__________________________________________________________________________________________________
conv2d_47 (Conv2D)              (None, 5, 5, 512)    1049088     activation_42[0][0]              
__________________________________________________________________________________________________
batch_normalization_47 (BatchNo (None, 5, 5, 512)    2048        conv2d_47[0][0]                  
__________________________________________________________________________________________________
activation_43 (Activation)      (None, 5, 5, 512)    0           batch_normalization_47[0][0]     
__________________________________________________________________________________________________
conv2d_48 (Conv2D)              (None, 5, 5, 512)    2359808     activation_43[0][0]              
__________________________________________________________________________________________________
batch_normalization_48 (BatchNo (None, 5, 5, 512)    2048        conv2d_48[0][0]                  
__________________________________________________________________________________________________
activation_44 (Activation)      (None, 5, 5, 512)    0           batch_normalization_48[0][0]     
__________________________________________________________________________________________________
conv2d_49 (Conv2D)              (None, 5, 5, 2048)   1050624     activation_44[0][0]              
__________________________________________________________________________________________________
batch_normalization_49 (BatchNo (None, 5, 5, 2048)   8192        conv2d_49[0][0]                  
__________________________________________________________________________________________________
add_14 (Add)                    (None, 5, 5, 2048)   0           activation_42[0][0]              
                                                                 batch_normalization_49[0][0]     
__________________________________________________________________________________________________
activation_45 (Activation)      (None, 5, 5, 2048)   0           add_14[0][0]                     
__________________________________________________________________________________________________
conv2d_50 (Conv2D)              (None, 5, 5, 512)    1049088     activation_45[0][0]              
__________________________________________________________________________________________________
batch_normalization_50 (BatchNo (None, 5, 5, 512)    2048        conv2d_50[0][0]                  
__________________________________________________________________________________________________
activation_46 (Activation)      (None, 5, 5, 512)    0           batch_normalization_50[0][0]     
__________________________________________________________________________________________________
conv2d_51 (Conv2D)              (None, 5, 5, 512)    2359808     activation_46[0][0]              
__________________________________________________________________________________________________
batch_normalization_51 (BatchNo (None, 5, 5, 512)    2048        conv2d_51[0][0]                  
__________________________________________________________________________________________________
activation_47 (Activation)      (None, 5, 5, 512)    0           batch_normalization_51[0][0]     
__________________________________________________________________________________________________
conv2d_52 (Conv2D)              (None, 5, 5, 2048)   1050624     activation_47[0][0]              
__________________________________________________________________________________________________
batch_normalization_52 (BatchNo (None, 5, 5, 2048)   8192        conv2d_52[0][0]                  
__________________________________________________________________________________________________
add_15 (Add)                    (None, 5, 5, 2048)   0           activation_45[0][0]              
                                                                 batch_normalization_52[0][0]     
__________________________________________________________________________________________________
activation_48 (Activation)      (None, 5, 5, 2048)   0           add_15[0][0]                     
__________________________________________________________________________________________________
global_average_pooling2d (Globa (None, 2048)         0           activation_48[0][0]              
__________________________________________________________________________________________________
dropout (Dropout)               (None, 2048)         0           global_average_pooling2d[0][0]   
__________________________________________________________________________________________________
dense (Dense)                   (None, 200)          409800      dropout[0][0]                    
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 200)          0           dense[0][0]                      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            201         dropout_1[0][0]                  
==================================================================================================
Total params: 23,991,441
Trainable params: 23,938,321
Non-trainable params: 53,120
__________________________________________________________________________________________________
```

**Result**

![image](https://user-images.githubusercontent.com/84713532/212814512-6bbee476-f37a-42dd-beeb-4cb2698903d6.png)

---

### Resnet50 keras model

![image](https://user-images.githubusercontent.com/84713532/212814650-b20ca9cf-bee7-444b-afa4-bf252bd01bc0.png)

```py
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose=1,factor=0.3, min_lr=0.000001)
```
```py
history = model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs = 25 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])
```

**Result**

![image](https://user-images.githubusercontent.com/84713532/212814750-02f3da90-f414-4b2e-bd5c-861a01061903.png)
