import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from matplotlib import image as mpimg

# Завантаження зображення
image_path = 'Amogus.png'
img = mpimg.imread(image_path)

# Перевірка діапазону значень зображення
if img.max() > 1:
    img = img / 255.0  # Нормалізація до [0, 1]

# Параметри для мультиплікативного шуму
a = 0.9
b = 1.1

# Створення усереднювального фільтру для заданого розміру
def create_average_filter(size):
    return np.ones((size, size)) / (size * size)

# Функція для фільтрації зображення
def apply_filter(image, filter_kernel):
    result = np.zeros_like(image)
    for channel in range(image.shape[2]):
        result[..., channel] = convolve2d(image[..., channel], filter_kernel, mode='same', boundary='symm')
    return result

# Додавання мультиплікативного шуму
def add_multiplicative_noise(image, a, b):
    noise = np.random.uniform(a, b, image.shape)
    noisy_image = image * noise
    return np.clip(noisy_image, 0, 1)

# Створення усереднювальних фільтрів
filter_3x3 = create_average_filter(3)
filter_5x5 = create_average_filter(5)
filter_7x7 = create_average_filter(7)
filter_11x11 = create_average_filter(11)

# Додавання шуму до зображення
multiplicative_noisy_img = add_multiplicative_noise(img, a, b)

# Фільтрація зображення з різними фільтрами
multiplicative_filtered_3x3 = apply_filter(multiplicative_noisy_img, filter_3x3)
multiplicative_filtered_5x5 = apply_filter(multiplicative_noisy_img, filter_5x5)
multiplicative_filtered_7x7 = apply_filter(multiplicative_noisy_img, filter_7x7)
multiplicative_filtered_11x11 = apply_filter(multiplicative_noisy_img, filter_11x11)

# Виведення результатів на одному екрані
plt.figure(figsize=(12, 12))

# Відображення результатів для кожного фільтра
plt.subplot(2, 2, 1)
plt.imshow(multiplicative_filtered_3x3)
plt.title('Filtered: 3x3')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(multiplicative_filtered_5x5)
plt.title('Filtered: 5x5')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(multiplicative_filtered_7x7)
plt.title('Filtered: 7x7')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(multiplicative_filtered_11x11)
plt.title('Filtered: 11x11')
plt.axis('off')

# Показати графік
plt.tight_layout()
plt.show()
