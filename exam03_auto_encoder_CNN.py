# autoencoder_02 모델의 CNN

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

#encoding
input_img = Input(shape = (28, 28, 1))                          # 1 = none color
x = Conv2D(16, (3,3), activation="relu", padding='same')(input_img)
x = MaxPool2D((2,2), padding="same")(x)
x = Conv2D(8, (3,3), activation="relu", padding='same')(x)
x = MaxPool2D((2,2), padding="same")(x)
x = Conv2D(8, (3,3), activation="relu", padding='same')(x)
encoded = MaxPool2D((2,2), padding="same")(x)

#decoding
x = Conv2D(8, (3,3), activation="relu", padding="same")(encoded)
x = UpSampling2D((2,2))(x)                                      # 이미지를 decoding 하는 과정이니 다시 늘림
x = Conv2D(8, (3,3), activation="relu", padding="same")(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation="relu")(x)                     # 16-(3-1)만큼 감소함
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

#모델 설계
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
conv_x_train = x_train.reshape(-1, 28, 28, 1)
conv_x_test = x_test.reshape(-1, 28, 28, 1)
# print(conv_x_train.shape)
# print(conv_x_test.shape)

# encoder를 학습
fit_hist = autoencoder.fit(conv_x_train, conv_x_train, epochs=50, batch_size=256, validation_data=(conv_x_test, conv_x_test))
decoded_img = autoencoder.predict(conv_x_test[:10])

# 학습 결과
n = 10
plt.gray()
plt.figure(figsize=(20,4))
for i in range(n) :
    ax = plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()