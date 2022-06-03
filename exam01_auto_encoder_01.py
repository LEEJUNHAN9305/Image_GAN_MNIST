# GAN에 진입하기 전
# AutoEncoder란? 입력값을 encoding 하여 출력값을 decoding 해줌

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

# 단순 오토인코딩 예시
input_img = Input(shape = (784, ))
encoded = Dense(32, activation="relu") # 784를 32로 압축
encoded = encoded(input_img)
decoded = Dense(784, activation="relu") # 다시 784로 늘어남
decoded = decoded(encoded)
autoencoder = Model(input_img, decoded)
# autoencoder.summary()

#인코더 정의
encoder = Model(input_img, encoded)
# encoder.summary()

#디코더 정의
encoded_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
# decoder.summary()

#모델 설계
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
flatted_x_train = x_train.reshape(-1, 28*28)
flatted_x_test = x_test.reshape(-1, 28*28)
# print(flatted_x_train.shape)
# print(flatted_x_test.shape)

# encoder를 학습
fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs=50, batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

encoded_img = encoder.predict(x_test[:10].reshape(-1, 28*28))
decoded_img = decoder.predict(encoded_img)

# 학습 결과
n = 10
plt.gray()
plt.figure(figsize=(20,4))
for i in range(n) :
    ax = plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False) # X축이 안 보임
    ax.get_yaxis().set_visible(False) # Y축도 안 보임 -- 그림만 보겠다

    ax = plt.subplot(2, 10, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()