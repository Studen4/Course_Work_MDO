import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

# Завантаження кольорового зображення
image_path = 'Amogus.png'
img = mpimg.imread(image_path)

# Перевірка діапазону значень зображення (іноді завантажується як [0, 1], іноді як [0, 255])
if img.max() > 1:
    img = img / 255.0  # Нормалізація до [0, 1]

# Параметри для рівномірного мультиплікативного шуму
var_number = 2 * 5
a = 1 - var_number / 100  # Мінімальне значення
b = 1 + var_number / 100  # Максимальне значення

# Генерація рівномірного мультиплікативного шуму
multiplicative_noise = np.random.uniform(a, b, img.shape)

# Додавання шуму
noisy_img = img * multiplicative_noise

# Обмеження значень у діапазоні [0, 1]
noisy_img = np.clip(noisy_img, 0, 1)

# Перетворення назад у діапазон [0, 255] для відображення
noisy_img = (noisy_img * 255).astype(np.uint8)

# Виведення результату
plt.figure(figsize=(6, 6))
plt.imshow(noisy_img)  # Кольорове зображення
plt.title('Multiplicative Noise')
plt.axis('off')
plt.show()
