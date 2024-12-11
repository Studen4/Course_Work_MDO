import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image

image_path = 'Amogus.png'
image = imread(image_path)

image = Image.fromarray((image * 255).astype(np.uint8))
image = image.resize((256, 256))
image = np.asarray(image) / 255.0

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
plt.show()

# Параметри гаусового шуму
variant_number = 2
m = 0
sigma = variant_number / 100

gaussian_noise = np.random.normal(m, sigma, image.shape)
noisy_image = image + gaussian_noise
noisy_image = np.clip(noisy_image, 0, 1)


plt.figure(figsize=(6, 6))
plt.imshow(noisy_image)
plt.title("Additive noise")
plt.axis('off')
plt.show()
