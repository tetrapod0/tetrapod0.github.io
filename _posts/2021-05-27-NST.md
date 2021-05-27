---
layout: post
title:  "Neural Style Transfer"
date:   2021-05-27
excerpt: "NST 구현해보기"
tag:
- NST
- Neural Style Transfer
comments: true
---

### 신경망 스타일 전이

말그대로 이미지의 스타일, 패턴을 다른이미지에 씌워준다.

예를 들어 사진에 반 고흐의 그림 스타일을 적용하여 새로운 그림을 만들 수 있다.

---

---

![고흐1](https://user-images.githubusercontent.com/48349693/119789272-e016c200-bf0d-11eb-8be1-cd7d897178e8.jpg)

고흐의 '별이 빛나는 밤'이다.

![밤3](https://user-images.githubusercontent.com/48349693/119789335-f15fce80-bf0d-11eb-92f2-575adbcb6262.jpg)

이 사진을 고흐그림처럼 만들 수 있다.

---

---

![nst](https://user-images.githubusercontent.com/48349693/119793018-2de0f980-bf11-11eb-98ea-e94ffa7a4899.PNG)

기본 구상은 위 그림과 같다

target image가 바뀔 이미지 이고 style image가 전이될 이미지이다.

분홍 박스가 여기서는 vgg 모델이 된다.

파란색 화살표가 목표할 방향을 나타낸다. 즉 가리키는것이 label이고 이전것이 yhat이라고 생각하면 된다.

여기선 모델을 학습시키는것이 아니고 이미지를 변환시킨다.

파란색 화살표에 의한 loss가 발생할텐데 loss에 대한 gradient는 이미지에 대해서만 구한다.

맨 밑 파란색화살표는 복제된 자신한테 가는데 이미지가 변형되면서 큰 틀은 유지하기 위해서이다.

---

---

### 필요한 모듈
```python
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from IPython import display
```

### 전역변수(상수)
```python
target_img_path = './밤3.jpg'
style_img_path = './고흐1.jpg'
IMG_MAX_SIZE = 512
style_weight = 1e-2
content_weight = 1e4
LEN_STYLE = 5
LAYERS = ['block1_conv1',
          'block2_conv1',
          'block3_conv1', 
          'block4_conv1', 
          'block5_conv1', 
          'block5_conv4']

epochs = 100
```

### 이미지 불러오기
```python
def load_img(img_path):
    img = tf.image.decode_jpeg(tf.io.read_file(img_path)) # uint8
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:2], tf.float32)
    scale = IMG_MAX_SIZE / max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
    new_img = tf.image.resize(img, new_shape) # float
    return new_img[tf.newaxis, ...]
```
불러오면서 형식이나 크기를 맞춘다.

### 이미지 확인
```python
target_img = load_img(target_img_path)
style_img = load_img(style_img_path)

plt.figure(figsize=(10,20))
plt.subplot(121)
plt.imshow(target_img[0])
plt.subplot(122)
plt.imshow(style_img[0])
plt.show()
```
![1](https://user-images.githubusercontent.com/48349693/119793839-ead35600-bf11-11eb-9aa4-5f211e144d2a.PNG)

### 모델 정의
```python
def vgg_layers():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in LAYERS]
    model = tf.keras.Model([vgg.input], outputs)
    return model
```
input에 대한 미리정한 layer들의 output을 가진 모델을 준다.

### output 처리 함수
```python
def gram_matrix(output1):
    res = tf.einsum('abcd,abce->ade', output1, output1)
    shape = tf.shape(output1)
    divisor = tf.cast(shape[1] * shape[2], tf.float32)
    return res[0] / divisor
```

이미지마다 크기가 다르기때문에 같은 layer인데도 shape이 다르다.

통일 시키기 위해 그 layer의 channel로 통일시킨다.

### label 만들기
```python
model = vgg_layers()

pre_style_img = tf.keras.applications.vgg19.preprocess_input(style_img * 255.0)
pre_target_img = tf.keras.applications.vgg19.preprocess_input(target_img * 255.0)
style_outputs = model(pre_style_img)
target_outputs = model(pre_target_img)

style_outputs = list(map(gram_matrix, style_outputs))
target_outputs = list(map(gram_matrix, target_outputs))

goal_outputs = style_outputs[:LEN_STYLE] + target_outputs[LEN_STYLE:]
```

vgg19를 사용하기때문에 vgg19용 전처리 함수를 사용한다.

goal_outputs가 라벨이 된다.

shape은 다음과 같다.

```python
for v in goal_outputs:
    print(v.shape)

(64, 64)
(128, 128)
(256, 256)
(512, 512)
(512, 512)
(512, 512)
```

### loss 함수 정의
```python
def get_loss(outputs, goal_outputs):
    style_loss = tf.add_n([tf.reduce_mean((out - goal)**2) for out, goal in zip(outputs[:LEN_STYLE], goal_outputs[:LEN_STYLE])])
    content_loss = tf.add_n([tf.reduce_mean((out - goal)**2) for out, goal in zip(outputs[LEN_STYLE:], goal_outputs[LEN_STYLE:])])
    
    style_loss *= style_weight / LEN_STYLE
    content_loss *= content_weight / (len(LAYERS) - LEN_STYLE)

    total_loss = style_loss + content_loss

    return total_loss
```
오차는 제곱오차평균이다.

style_loss는 5개를 더한것이기때문에 5로 나눈다. 여기서 LEN_STYLE은 5이다.

### train step
```python
target_img = tf.Variable(target_img)

optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

def train_step():
    with tf.GradientTape() as tape:
        tape.watch(target_img)
        pre_target_img = tf.keras.applications.vgg19.preprocess_input(target_img * 255.0)
        outputs = model(pre_target_img)
        outputs = list(map(gram_matrix, outputs))

        loss = get_loss(outputs, goal_outputs)
        grad = tape.gradient(loss, target_img)
        optimizer.apply_gradients([(grad, target_img)])
        target_img.assign(tf.clip_by_value(target_img, 0.0, 1.0))
```
gradient를 이미지에 적용시키기위해 tensor 변수를 사용한다.

grad가 잘 전달되는 것을 보니 vgg19 전처리도 tensor만을 사용하나보다.

### 실행
```python
def run():
    start = time.time()
    step = 0
    for epoch in range(epochs):
      for m in range(50):
        step += 1
        train_step()
        print(".", end='')
      display.clear_output(wait=True)
      plt.figure(figsize=(10,10))
      plt.imshow(target_img[0])
      plt.axis('off')
      plt.savefig('./img/image_at_epoch_{:04d}.png'.format(epoch+1))
      plt.show()
      
      print("Train step: {}".format(step))
    end = time.time()
    print("Total time: {:.1f}".format(end-start))
```
50번마다 이미지를 출력하기위해 50번을 한 epoch로 두었다.

100 epoch의 총 실행시간은 10분 가량 걸렸다.

![nst](https://user-images.githubusercontent.com/48349693/119796887-b8772800-bf14-11eb-9834-c61a342669a3.gif)

---

---

참고URL : <https://www.tensorflow.org/tutorials/generative/style_transfer?hl=ko>
























