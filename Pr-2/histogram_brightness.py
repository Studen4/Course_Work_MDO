import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

image_path = 'Amogus.png'
img = mpimg.imread(image_path)

gray_img = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]) * 255
gray_img = gray_img.astype(np.uint8)

I_max = np.amax(gray_img)
I_min = np.amin(gray_img)

K = (I_max - I_min) / 255

# Побудова гістограми яскравості
M = 10
n, bins = np.histogram(gray_img.flatten(), bins=M, range=(0, 255))

plt.subplot(2, 1, 1)
plt.bar(bins[:-1], n, width=(bins[1] - bins[0]), color='gray', alpha=0.7, edgecolor='black')
plt.title('Histogram of Brightness Distribution')
plt.xlabel('Brightness (0-255)')
plt.ylabel('Number of Pixels')

# Розрахунок та побудова кумулятивної функції
delta = bins[1] - bins[0]  # Ширина інтервалу
H = n / (delta * gray_img.size)  # Густина розподілу (за формулою з вашого завдання)
Fk = np.cumsum(H * delta)  # Кумулятивна функція розподілу

plt.subplot(2, 1, 2)
plt.plot(bins[:-1], Fk, marker='o', color='blue', label='Cumulative Distribution Function')
plt.title('Cumulative Function of Brightness Distribution')
plt.xlabel('Brightness (0-255)')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.tight_layout()
plt.show()
