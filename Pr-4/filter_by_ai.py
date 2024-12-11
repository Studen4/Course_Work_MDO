import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.client import device_lib

variant = 2
sigma = ((variant / 100) ** 0.5)

# Завантаження та нормалізація даних
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Додавання шуму
x_train_noisy = x_train + np.random.normal(loc=0.0, scale=sigma, size=x_train.shape)
x_test_noisy = x_test + np.random.normal(loc=0.0, scale=sigma, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Побудова моделі з меншою кількістю фільтрів
input_image = keras.layers.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_image)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
encoded = x
# Декодування
x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = keras.layers.UpSampling2D((2, 2))(x)  # Відновлення розміру
decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

model = keras.models.Model(input_image, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(
    x_train_noisy, x_train,
    epochs=10,
    batch_size=32,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

model.save('filter_model_optimized.keras')
model = keras.models.load_model('filter_model_optimized.keras')

reconstructed_images = model.predict(x_test_noisy)

indices = [variant, variant + 500, variant + 1000, variant + 1500]
plt.figure(figsize=(10, 10))
for i, idx in enumerate(indices):

    # Початкові зображення
    plt.subplot(4, 3, i * 3 + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Зашумлені зображення
    plt.subplot(4, 3, i * 3 + 2)
    plt.imshow(x_test_noisy[idx].reshape(28, 28), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

    # Відфільтровані зображення
    plt.subplot(4, 3, i * 3 + 3)
    plt.imshow(reconstructed_images[idx].reshape(28, 28), cmap='gray')
    plt.title("Denoised")
    plt.axis('off')

plt.tight_layout()
plt.show()
