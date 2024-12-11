import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

image_path = 'Amogus.png'
img = mpimg.imread(image_path)

# Перетворення зображення у напівтонове (grayscale)
if len(img.shape) == 3:
    gray_img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
else:
    gray_img = img

i_min = np.min(gray_img)
i_max = np.max(gray_img)

# Лінійне розтягнення, щоб L(min) = 0, L(max) = 255
stretched_img = (gray_img - i_min) * 255 / (i_max - i_min)  # Масштабуємо значення в діапазон [0, 255]
stretched_img = np.clip(stretched_img, 0, 255).astype(np.uint8)  # обмежуємо значення в межах [0, 255]


# Функція для побудови гістограми
def plot_histogram(image, ax):
    intensities, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
    ax.bar(bins[:-1], intensities, width=1, color='gray')
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Pixel Count')


# Функція для побудови кумулятивної функції
def plot_cdf(image_cdf, ax):
    histogram, bins = np.histogram(image_cdf.flatten(), bins=256, range=(0, 256))
    cdf = np.cumsum(histogram)
    cdf_normalized = cdf / cdf.max()  # Нормалізація
    ax.plot(cdf_normalized, color='blue')
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Normalized CDF')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Гістограма після лінійного розтягнення
plot_histogram(stretched_img, ax1)
ax1.set_title('Stretched Histogram')

# Кумулятивна функція після лінійного розтягнення
plot_cdf(stretched_img, ax2)
ax2.set_title('Stretched Cumulative Distribution')

plt.tight_layout()
plt.show()
