import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

image_path = 'Amogus.png'
img = mpimg.imread(image_path)

if img.max() > 1:
    img = img / 255.0  # Нормалізація до [0, 1]

# Перевірка кількості каналів (RGB або RGBA)
if img.shape[2] == 4:
    has_alpha = True
else:
    has_alpha = False

# Параметри для шуму типу "сіль і перець"
var_number = 2
d = 0.1 + abs(var_number - 14) / 100  # Розрахунок щільності шуму

# Копія зображення для додавання шуму
noisy_img = img.copy()

# Перебір кожного пікселя і додавання шуму
for i in range(img.shape[0]):  # Перебір по висоті (рядки)
    for j in range(img.shape[1]):  # Перебір по ширині (стовпці)
        # Генерація випадкового числа для кожного пікселя
        rand_val = np.random.rand()

        if rand_val < d:
            # Сіль (чорний піксель)
            noisy_img[i, j, :3] = [0, 0, 0]  # Встановлюємо чорний піксель (ігноруємо альфа-канал)
        elif rand_val > (1 - d):
            # Перець (білий піксель)
            noisy_img[i, j, :3] = [1, 1, 1]  # Встановлюємо білий піксель (ігноруємо альфа-канал)

noisy_img = (noisy_img * 255).astype(np.uint8)

plt.figure(figsize=(6, 6))
plt.imshow(noisy_img)
plt.title('Salt & Pepper Noise')
plt.axis('off')
plt.show()
