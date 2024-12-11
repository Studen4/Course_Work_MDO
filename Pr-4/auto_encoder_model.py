import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


class AutoencoderModel:
    def __init__(self):
        self.model = None

    def build_model(self, input_shape=(28, 28, 1)):
        input_image = Input(shape=input_shape)

        # Енкодер
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)
        x = MaxPooling2D((2, 2), padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # Декодер
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.model = Model(input_image, decoded)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy')

    def train_model(self, x_train_noisy, x_train, x_test_noisy, x_test, epochs=100, batch_size=128):
        self.model.fit(x_train_noisy, x_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_data=(x_test_noisy, x_test))

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        from keras.models import load_model
        self.model = load_model(filename)

    def test_model(self, x_test_noisy):
        return self.model.predict(x_test_noisy)

    def plot_model_structure(self, filename='model_structure.png'):
        if self.model:
            plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)
            print(f"Структуру моделі збережено у файл '{filename}'.")
        else:
            print("Модель ще не побудована. Виконайте метод build_model() перед візуалізацією.")
