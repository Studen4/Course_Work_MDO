import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

# Завантаження зображення
image_path = 'Amogus.png'
img = mpimg.imread(image_path)

# Перетворення зображення у напівтонове (grayscale)
if len(img.shape) == 3:
    gray_img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
else:
    gray_img = img


def cumulative_distribution(histogram):
    cum_sum = np.cumsum(histogram)
    return cum_sum / cum_sum.max()


# Гістограма та CDF початкового зображення
original_histogram, original_bins = np.histogram(gray_img.flatten(), bins=256, range=(0, 256))
original_cdf = cumulative_distribution(original_histogram)

# Лінійне розтягування
i_min = np.min(gray_img)
i_max = np.max(gray_img)
k = i_max - i_min

stretched_img = ((gray_img - i_min) / k * 255).clip(0, 255).astype(np.uint8)

# Гістограма та CDF після лінійного розтягування
stretched_histogram, stretched_bins = np.histogram(stretched_img.flatten(), bins=256, range=(0, 256))
stretched_cdf = cumulative_distribution(stretched_histogram)

# Рівномірний розподіл (Equalization)
cdf_min = stretched_cdf.min()
cdf = (stretched_cdf - cdf_min) / (1 - cdf_min) * 255
equalized_img = cdf[stretched_img]

# Гістограма та CDF після рівномірного розподілу
equalized_histogram, equalized_bins = np.histogram(equalized_img.flatten(), bins=256, range=(0, 256))
equalized_cdf = cumulative_distribution(equalized_histogram)

# Обчислення статистики еквалізованого зображення
eq_i_min = np.min(equalized_img)
eq_i_max = np.max(equalized_img)
eq_k = eq_i_max - eq_i_min
print(f"Equalized Image Statistics: Imin = {eq_i_min}, Imax = {eq_i_max}, K = {eq_k}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Гістограма
ax1.bar(equalized_bins[:-1], equalized_histogram, width=1, color='gray', alpha=0.7)
ax1.set_title('Equalized Histogram')
ax1.set_ylabel('Pixel Count')

# Кумулятивна функція
ax2.plot(equalized_cdf, color='blue')
ax2.set_title('Equalized Cumulative Distribution')
ax2.set_xlabel('Intensity Value')
ax2.set_ylabel('Normalized CDF')

plt.tight_layout()
plt.show()

# Відображення еквалізованого зображення
plt.imshow(equalized_img, cmap='gray')
plt.title('Equalized Image, Lmax=255, Lmin=0, K=1.0')
plt.axis('off')
plt.show()
