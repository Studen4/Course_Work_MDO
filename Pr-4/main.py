from preprocessor_data import DataPreprocessor
from adder_noise import NoiseAdder
from auto_encoder_model import *
import matplotlib.pyplot as plt


def main():
    # Завантаження та підготовка даних
    x_train, x_test = DataPreprocessor.load_and_preprocess_data()

    # Додавання шуму
    variance = 2  # Ваш варіант
    noise_adder = NoiseAdder(variance)
    x_train_noisy = noise_adder.add_noise(x_train)
    x_test_noisy = noise_adder.add_noise(x_test)

    # Побудова та навчання моделі
    autoencoder = AutoencoderModel()
    autoencoder.build_model()
    autoencoder.train_model(x_train_noisy, x_train, x_test_noisy, x_test)

    # Збереження моделі
    autoencoder.save_model('filter_model.keras')

    # Перевірка моделі
    reconstructed_images = autoencoder.test_model(x_test_noisy)

    indices = [2, 502, 1002, 1502]
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(3, 4, i + 5)
        plt.imshow(x_test_noisy[idx].reshape(28, 28), cmap='gray')
        plt.title("Noisy")
        plt.axis('off')

        plt.subplot(3, 4, i + 9)
        plt.imshow(reconstructed_images[idx].reshape(28, 28), cmap='gray')
        plt.title("Denoised")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
