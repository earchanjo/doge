import cv2
from cv2 import threshold 
import numpy as np

screen_img = cv2.imread("screen.png", cv2.IMREAD_UNCHANGED)
doge_img = cv2.imread("doge_avatar.png", cv2.IMREAD_UNCHANGED)



result = cv2.matchTemplate(screen_img, doge_img, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(max_val)

w = doge_img.shape[1]
h = doge_img.shape[0]
cv2.rectangle(screen_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0,255,255), 1)

threshold = 1

yloc, xloc = np.where(result >= threshold)
for (x, y) in zip(xloc, yloc):
    cv2.rectangle(screen_img, (x, y), (x + w, y + h), (0,255,255), 2)


rectangles = []
for (x, y) in zip(xloc, yloc):
    rectangles.append([int(x), int(y), int(w), int(h)])
    rectangles.append([int(x), int(y), int(w), int(h)])

print(len(rectangles))
rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)
print(len(rectangles))

cv2.imshow('Result', screen_img)
cv2.waitKey()
cv2.destroyAllWindows()