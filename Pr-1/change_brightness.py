import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

image_path = 'Amogus.png'
img = mpimg.imread(image_path)

gray_img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

I_max = np.amax(gray_img)
I_min = np.amin(gray_img)

low_out = 15
high_out = 195
Y = 1

# Степеневе перетворення
ImS = low_out + (high_out - low_out) * ((gray_img - I_min) / (I_max - I_min)) ** Y
I_min_new = np.amin(ImS)
I_max_new = np.amax(ImS)
K_new = (I_max_new - I_min_new) / 255


plt.imshow(ImS, cmap='gray')
plt.title(f'Changed, Imax = {I_max_new:.2f}, Imin = {I_min_new:.2f}, K = {K_new:.2f}')
plt.axis('off')
plt.show()
