---
layout: post
title:  "Captcha OCR"
date:   2021-09-05
excerpt: "captcha 글자이미지 해독하기"
tag:
- captcha
- ctc
- OCR
comments: true
---

### captcha 글자 해독하기

---

```python
!wget https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
!unzip -q captcha_images_v2.zip
```

captcha 이미지를 다운해준다.

```python
import os
from glob import glob

img_paths = sorted(glob('./captcha_images_v2/*.png'))
get_label = lambda x:os.path.splitext(os.path.basename(x))[0]
labels = list(map(get_label, img_paths))
chars = set(''.join(labels))
max_length = max([len(label) for label in labels])

print("Number of images found: ", len(img_paths))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(chars))
print("Characters present: ", chars)
```

이미지경로와 라벨 얻기

어떤 글자가 쓰였는지도 필요하다.

결과는 다음과 같다.

```
Number of images found:  1040
Number of labels found:  1040
Number of unique characters:  19
Characters present:  {'x', 'd', '2', '8', 'b', 'w', 'p', 'n', 'y', 'c', 'e', '4', 'f', '5', 'g', '7', '3', '6', 'm'}
```

```python
import matplotlib.pyplot as plt

img = plt.imread(img_paths[0])
label = labels[0]

plt.title(label)
plt.imshow(img)
```

테스트로 이미지를 확인해본다.

![다운로드](https://user-images.githubusercontent.com/48349693/132122370-16abe82d-9cc6-40f6-b0ae-cf199a4efa85.png)

```python
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(img_paths, labels, test_size=0.1, random_state=2021)

print(len(x_train), len(y_train))
print(len(x_valid), len(y_valid))
```
split data

```python
import tensorflow as tf
from tensorflow.keras import layers

# layers.experimental.preprocessing.StringLookup

# Mapping characters to integers
char_to_num = layers.StringLookup(
    vocabulary=list(chars), mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

img_width = 200
img_height = 50

def encode_single_sample(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32) # 0~255 -> 0~1
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

    return img, label
```
데이터 전처리함수

```python
import tensorflow as tf

batch_size = 16

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
```
Dataset을 이용하여 train_dataset과 validation_dataset을 만든다.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

loss_fn = keras.backend.ctc_batch_cost

def CTC_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64") # 16
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64") # 50
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64") # 5

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = loss_fn(y_true, y_pred, input_length, label_length)
    return loss

def build_model():
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image", dtype="float32")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", 
                      kernel_initializer="he_normal", name="Conv2D_1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="Pool_1")(x) # (100, 25, 32)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                      kernel_initializer="he_normal", name="Conv2D_2")(x)
    x = layers.MaxPooling2D((2, 2), name="Pool_2")(x) # (50, 12, 64)

    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="Reshape")(x) # (50, 768)
    # x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Conv1D(64, 1, activation='relu', name='Conv1D_1')(x)
    x = layers.Dropout(0.2, name='Dropout')(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25), name='Bi_LSTM_1')(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25), name='Bi_LSTM_2')(x)

    # Output layer
    # x = layers.Dense(len(char_to_num.get_vocabulary()) + 1,
    #                  activation="softmax", name="dense2")(x)
    x = layers.Conv1D(len(char_to_num.get_vocabulary()) + 1, 1, activation='softmax', name='Conv1D_2')(x)

    # Define the model
    model = keras.models.Model(inputs=input_img, outputs=x, name="ocr_model_v1")

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=CTC_loss)

    return model


# Get the model
model = build_model()
model.summary()
```

모델생성함수부분이다.

원본코드에서는 Dense layer를 사용하였지만

생각해보면 Conv1D layer로 kernel_size를 1로만 하면 같은 연결이 만들어진다.

원본코드의 CTC_layer도 뺐고 따로 CTC_loss함수로 만들었는데 

왜 복잡하게 모델안에서 loss를 처리하려했는지는 모르겠다.

```python
epochs = 100
early_stopping_patience = 10
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)
```

원본코드와 동일하게 학습시켰다.

그런데 학습과정이 좀 특이하다.

![다운로드](https://user-images.githubusercontent.com/48349693/132122661-3e8c9c77-9ea7-4da1-9921-f8513ca32e25.png)

이런 그래프는 처음본다.

```python
from tensorflow.keras import layers
from tensorflow import keras

class CTCDecodeLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(CTCDecodeLayer, self).__init__(**kwargs)
        self.decoder = keras.backend.ctc_decode
    
    def __call__(self, y_pred):
        shape = tf.shape(y_pred)
        input_len = tf.ones(shape[0]) * tf.cast(shape[1], dtype=tf.float32)
        results = self.decoder(y_pred, input_length=input_len, greedy=True)[0][0][:,:max_length]
        return tf.strings.reduce_join(num_to_char(results), axis=1)

image = layers.Input(shape=(img_width, img_height, 1), name='Image')
y_pred = model(image, training=False)
decoded = CTCDecodeLayer(name='CTC_Decode')(y_pred)
inference_model = tf.keras.Model(inputs=image, outputs=decoded)

inference_model.summary()
```

image to string 모델이다.

이미지를 넣으면 문자열로 바로나오게 inference_model을 만들었다.

```python
import numpy as np

for batch in validation_dataset.take(1):
    batch_images, batch_labels = batch

    batch_preds = inference_model.predict(batch_images)
    batch_labels = tf.strings.reduce_join(num_to_char(batch_labels), axis=1).numpy()

    batch_preds = list(map(lambda x:x.decode('utf-8'), batch_preds))
    batch_labels = list(map(lambda x:x.decode('utf-8'), batch_labels))

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(batch_preds)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {batch_preds[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
plt.show()
```

![캡처](https://user-images.githubusercontent.com/48349693/132122779-484f2f67-cbdb-4146-bba2-694b90d11af5.PNG)

매우 정확하게 나온것 같다.

---

참고 URL : 

<https://keras.io/examples/vision/captcha_ocr/>

<https://hulk89.github.io/machine%20learning/2018/01/30/ctc/>

<https://www.kaggle.com/fournierp/captcha-version-2-images>
