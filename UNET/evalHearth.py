import os
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

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("resultsHearth")
    create_dir("AiMask")
    create_dir("MyMask")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/modelHearth.h5")

    """ Dataset """
    dataset_path = "MontgomerySet"
    (train_x, train_y1), (valid_x, valid_y1), (test_x, test_y1) = load_data(dataset_path)

    """ Predicting the mask """
    for x, y1 in tqdm(zip(test_x, test_y1), total=len(test_x)):
        """ Extracing the image name. """
        image_name = "0c27c918-MCUCXR_0054_0.png"
        filename = os.path.basename(image_name)

        """ Reading the image """
        ori_x = cv2.imread(x, cv2.IMREAD_COLOR)
        ori_x = cv2.resize(ori_x, (W, H))
        x = ori_x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Reading the mask """
        ori_y1 = cv2.imread(y1, cv2.IMREAD_GRAYSCALE)
        ori_y = ori_y1
        ori_y = cv2.resize(ori_y, (W, H))
        ori_y = np.expand_dims(ori_y, axis=-1)  ## (512, 512, 1)
        ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)  ## (512, 512, 3)
        cv2.imwrite(f'MyMask/{filename}', ori_y)

        """ Predicting the mask. """
        y_pred = model.predict(x)[0] > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the predicted mask along with the image and GT """
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)

        plt.imshow(y_pred*255)
        plt.show()

        cv2.imwrite(f'AiMask/{filename}', y_pred*255)


        sep_line = np.ones((H, 10, 3)) * 255

        cat_image = np.concatenate([ori_x, sep_line, ori_y, sep_line, y_pred*255], axis=1)

        cv2.imwrite(f'resultsHearth/{filename}', cat_image)


