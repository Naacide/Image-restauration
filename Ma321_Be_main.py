# -*- coding: utf-8 -*-

"""
Main script for testing image restoration

Example image files available (place them in the same folder as this script):
- man_bw.jpg      # Grayscale
- man_color.jpg    # Color


@author: Nathan_AZO
"""

import cv2
from Ma321_Be_lib import restauration, restauration_couleur, masque

# --- Load an example image ---
F = cv2.imread('man_bw.jpg', cv2.IMREAD_UNCHANGED)  # change filename as needed

if F is None:
    raise FileNotFoundError("Image file not found. Check the filename and path.")

# --- Create mask ---
M = masque(F, 0)

# --- Detect type and restore ---
if len(F.shape) == 2 or F.shape[2] == 1:
    # Grayscale restoration
    restored = restauration(F, M, epsilon=1)
    cv2.imshow("Restored Grayscale Image", restored)
else:
    # Color restoration
    restored = restauration_couleur(F, M, epsilon=1)
    cv2.imshow("Restored Color Image", restored)

cv2.waitKey(0)
cv2.destroyAllWindows()
