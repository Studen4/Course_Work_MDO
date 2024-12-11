import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from matplotlib import image as mpimg

# Завантаження зображення
image_path = 'Amogus.png'
img = mpimg.imread(image_path)

# Перевірка діапазону значень зображення
if img.max() > 1:
    img = img / 255.0  # Нормалізація до [0, 1]

# Параметри для шуму
var_number = 2
a = 0.9
b = 1.1
d = 0.1 + abs(var_number - 14) / 100  # Для "сіль і перець"

# Функція для додавання шуму
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

    # Додаємо сіль
    noisy_image[salt_and_pepper_noise < d] = 0

    # Додаємо перець
    noisy_image[salt_and_pepper_noise > (1 - d)] = 1

    return noisy_image

# Додавання шуму до зображень
additive_noisy_img = add_additive_noise(img, 0, var_number / 100)  # адитивний гаусів шум
multiplicative_noisy_img = add_multiplicative_noise(img, a, b)  # мультиплікативний шум
salt_and_pepper_noisy_img = add_salt_and_pepper_noise(img, d)  # шум сіль і перець

# Фільтрація за допомогою медіанного фільтру
median_filtered_5x5 = median_filter(additive_noisy_img, size=5)
median_filtered_7x7 = median_filter(multiplicative_noisy_img, size=7)
median_filtered_11x11 = median_filter(salt_and_pepper_noisy_img, size=11)

# Виведення результатів на одному екрані
plt.figure(figsize=(12, 12))

# Відображення оригіналу
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('а. Оригінал')
plt.axis('off')

# Відображення результатів
plt.subplot(2, 2, 2)
plt.imshow(median_filtered_5x5)
plt.title('б. Адитивний шум, фільтр 5x5')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(median_filtered_7x7)
plt.title('в. Мультиплікативний шум, фільтр 7x7')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(median_filtered_11x11)
plt.title('г. Сіль і перець, фільтр 11x11')
plt.axis('off')

# Показати графік
plt.tight_layout()
plt.show()