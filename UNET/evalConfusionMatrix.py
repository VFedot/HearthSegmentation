import os

from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
from train import load_data, create_dir, tf_dataset
import matplotlib.pyplot as plt

H = 512
W = 512


def load_images(path):
    images = sorted(glob(os.path.join(path, "ConfusionMatrix", "Images",  "*.png")))


    train_x = train_test_split(images)


    return (train_x)


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("resultsConfusion")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/modelHearth.h5")

    """ Dataset """
    dataset_path = "MontgomerySet"
    test_x = load_images(dataset_path)

    """ Predicting the mask """
    for sublist in test_x:
        for x in tqdm(sublist, total=len(sublist)):
            """ Extracing the image name. """
            image_name = x.split("/")[-1]

            """ Reading the image """
            ori_x = cv2.imread(x, cv2.IMREAD_COLOR)
            ori_x = cv2.resize(ori_x, (W, H))
            x = ori_x/255.0
            x = x.astype(np.float32)
            x = np.expand_dims(x, axis=0)

            """ Predicting the mask. """
            y_pred = model.predict(x)[0] > 0.5
            y_pred = y_pred.astype(np.int32)

            """ Saving the predicted mask along with the image and GT """
            filename = os.path.basename(image_name)
            y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)

            sep_line = np.ones((H, 10, 3)) * 255

            cat_image = np.concatenate([ori_x, sep_line, sep_line, y_pred*255], axis=1)

            cv2.imwrite(f'resultsConfusion/{filename}', cat_image)


