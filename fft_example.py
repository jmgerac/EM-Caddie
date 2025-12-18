import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(r'assets/caddie_img.jpg', cv2.IMREAD_GRAYSCALE)

# Compute 2D FFT and shift zero frequency to center
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# Plot original and FFT magnitude
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Input Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('FFT Magnitude Spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
