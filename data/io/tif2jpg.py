# -*-encoding:utf-8-*-
import os
import cv2

input_path = '/yangxue/data_ship_clean/VOCdevkit/TIFImages/'
output_path = '/yangxue/data_ship_clean/VOCdevkit/JPEGImages/'

images = os.listdir(input_path)

for count, i in enumerate(images):
    img = cv2.imread(os.path.join(input_path, str(i)))
    cv2.imwrite(os.path.join(output_path, i.replace('.tif', '.jpg')), img)
    if count % 1000 == 0:
        print(count)
