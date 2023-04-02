import os

from matplotlib import pyplot as plt

from UNET.metrics import iou, dice_coef, dice_loss
from PIL import Image, ImageFilter, ImageTk

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import tensorflow as tf
from keras.utils import CustomObjectScope

H = 512
W = 512

def segmentImage(image):
    plt.imshow(image, cmap='gray')
    plt.show()

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/modelHearth.h5")

        """ Reading the image """
        ori_x = cv2.resize(image, (W, H))
        x = ori_x/255.0
        x = x.astype(np.float32)
        x = np.stack((x,) * 3, axis=-1)
        x = np.expand_dims(x, axis=0)  # Add batch dimension

        """ Predicting the mask. """
        y_pred = model.predict(x)[0] > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the predicted mask along with the image and GT """
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
        mask = y_pred * 255
        mask = mask.astype(np.uint8)

        ori_x = np.stack((ori_x,) * 3, axis=-1)


        """ Selecting only the hearth region in the original image """
        heart_pixels = ori_x.copy()
        heart_pixels[np.where(mask == 0)] = 0
        plt.imshow(heart_pixels)
        plt.show()

        return heart_pixels
