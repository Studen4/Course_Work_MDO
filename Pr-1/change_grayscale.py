import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

image_path = 'Amogus.png'
img = mpimg.imread(image_path)

print("Розміри зображення:", img.shape)

# Визначення яскравості кожного пікселя за формулою I(ij) = 0.299 * Red + 0.587 * Green + 0.114 * Blue
gray_img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

# Обчислення максимального та мінімального значення яскравості
I_max = np.amax(gray_img)
I_min = np.amin(gray_img)

# Обчислення контрасту
I_star_max = 255
K = (I_max - I_min) / I_star_max

plt.imshow(gray_img, cmap='gray')
plt.title(f'Grayscale, K = {K:.3f}')
plt.axis('off')
plt.show()
