import numpy as np
import matplotlib.pyplot as plt
from change_brightness import ImS

# Перетворення зображення в негативне
I_max = 255  # Максимальна яскравість для 8-бітного зображення
ImN = I_max - ImS  # Негативне зображення

I_min_N = np.amin(ImN)
I_max_N = np.amax(ImN)

K_N = (I_max_N - I_min_N) / I_max

plt.imshow(ImN, cmap='gray')
plt.title(f'Negative, Imax = {I_max_N:.2f}, Imin = {I_min_N:.2f}, K = {K_N:.2f}')
plt.axis('off')
plt.show()
