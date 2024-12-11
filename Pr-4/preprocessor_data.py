from auto_encoder_model import *


class DataPreprocessor:
    @staticmethod
    def load_and_preprocess_data():
        (x_train, _), (x_test, _) = mnist.load_data()

        # Нормалізація
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        # Зміна форми
        x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
        x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

        return x_train, x_test
