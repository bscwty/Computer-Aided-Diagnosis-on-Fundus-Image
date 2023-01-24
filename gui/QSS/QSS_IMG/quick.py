import numpy as np
import cv2

img1 = cv2.imread('./icon.png')
img2 = cv2.imread('./icon.png', cv2.IMREAD_GRAYSCALE)

b, g, r = cv2.split(img1)
a = np.ones_like(b, dtype=b.dtype) * 255

space = np.where(img2>220)
a[space] = 0

img1 = cv2.merge((b, g, r, a))

cv2.imwrite('southeast.png', img1)