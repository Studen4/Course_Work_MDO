import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from filter_averaging import apply_filter, create_average_filter
from filter_median import median_filter

image_path = 'Amogus.png'
img = mpimg.imread(image_path)

if img.max() > 1:
    img = img / 255.0  # Нормалізація до [0, 1]


# Реалізації шумів
def add_additive_noise(image, m, sigma):
    noise = np.random.normal(m, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def add_multiplicative_noise(image, a, b):
    noise = np.random.uniform(a, b, image.shape)
    noisy_image = image * noise
    return np.clip(noisy_image, 0, 1)


def add_salt_and_pepper_noise(image, d):
    noisy_image = image.copy()
    salt_and_pepper_noise = np.random.rand(*image.shape[:2])
    noisy_image[salt_and_pepper_noise < d] = 0
    noisy_image[salt_and_pepper_noise > (1 - d)] = 1

    return noisy_image


# Параметри шумів
var_number = 2
a = 0.9
b = 1.1
d = 0.1 + abs(var_number - 14) / 100
filter_sizes = [3, 5, 7, 11]

# Генерація шумів
additive_noisy_img = add_additive_noise(img, 0, var_number / 100)  # Адитивний гаусів шум
multiplicative_noisy_img = add_multiplicative_noise(img, a, b)  # Мультиплікативний шум
salt_and_pepper_noisy_img = add_salt_and_pepper_noise(img, d)  # Шум "сіль і перець"

# Список шумів
noisy_images = {
    "Адитивний шум": additive_noisy_img,
    "Мультиплікативний шум": multiplicative_noisy_img,
    "Шум 'сіль і перець'": salt_and_pepper_noisy_img,
}


def visualize_filters(noisy_images, filter_type, filter_func, filter_name):
    for noise_name, noisy_img in noisy_images.items():
        # Створення фільтрів різного розміру
        filtered_images = []
        for size in filter_sizes:
            if filter_type == "averaging":
                filter_kernel = create_average_filter(size)
                filtered_img = apply_filter(noisy_img, filter_kernel)
            elif filter_type == "median":
                filtered_img = filter_func(noisy_img, size=size)
            filtered_images.append(filtered_img)

        plt.figure(figsize=(10, 10))
        plt.suptitle(f'{noise_name} ({filter_name})', fontsize=16)

        plt.subplot(2, 2, 1)
        plt.imshow(filtered_images[0])
        plt.title(f"Фільтр {filter_sizes[0]}x{filter_sizes[0]}")
        plt.axis("off")

        for i in range(1, len(filtered_images)):
            plt.subplot(2, 2, i + 1)
            plt.imshow(filtered_images[i])
            plt.title(f"Фільтр {filter_sizes[i]}x{filter_sizes[i]}")
            plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


visualize_filters(noisy_images, "averaging", None, "Усереднювальний фільтр")

visualize_filters(noisy_images, "median", median_filter, "Медіанний фільтр")
