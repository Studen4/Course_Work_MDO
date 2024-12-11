import numpy as np


class NoiseAdder:
    def __init__(self, variance):
        self.sigma = (variance / 100) ** 0.5

    def add_noise(self, images):
        noisy_images = images + np.random.normal(loc=0.0, scale=self.sigma, size=images.shape)
        return np.clip(noisy_images, 0., 1.)
