import config
from gan import GAN
import cv2

generator = GAN(buff_size=10035, batch_size=16, epochs=5000, imgs_size=(config.width, config.heigth))

while True:
    cv2.imshow("pupa", generator.generate_image())
    cv2.waitKey(0)