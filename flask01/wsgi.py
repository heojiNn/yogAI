import cv2
import numpy as np

scale_factor = 5
font = cv2.FONT_HERSHEY_TRIPLEX
fontScale = 1
fontColor = (255, 255, 255)
fontColor_dead = (0, 0, 255)
thickness = 1  # smallest possible (another solution could be rendering a resized image with the data)
lineType = 1
pos = .05
hd = (720, 1280, 3)
hdd = (1080, 1920, 3)
image = np.ones(hdd)
(h, w) = image.shape[:2]
left = image[:h, :w // 2]
right = image[:h, w // 2:]
left = np.zeros((left.shape[0], left.shape[1], 3))
left = cv2.putText(left,
                    "YOU DIED",
                    (int(left.shape[1] / 3), int(left.shape[0] / 2 * .9)),
                    font,
                    fontScale * 2,
                    fontColor_dead,
                    thickness * 2,
                    lineType)
image = np.concatenate((left, right), axis=1)
# draw line to visualize separator
image = cv2.line(image, (w // 2, 0), (w // 2, h), (0, 0, 0), 1)
cv2.imshow('image',image)
cv2.waitKey(0)
