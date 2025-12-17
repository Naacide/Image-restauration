# Image Restoration Project

This project implements image restoration techniques for grayscale and color images, aiming to reconstruct missing or corrupted pixels.

## Features

- Grayscale image restoration using Gradient Projection Conjugate (GPC) method
- Color image restoration by processing channels separately
- Masking of specific pixel values

## Example Usage

```python
import cv2
from lib import restauration, restauration_couleur, masque

# Load example image
F = cv2.imread('images/orig1_be.png', cv2.IMREAD_UNCHANGED)
M = masque(F, 0)

# Restore image
if len(F.shape) == 2 or F.shape[2] == 1:
    restored = restauration(F, M, epsilon=1)
else:
    restored = restauration_couleur(F, M, epsilon=1)

cv2.imshow("Restored Image", restored)
cv2.waitKey(0)
cv2.destroyAllWindows()
