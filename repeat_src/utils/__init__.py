from execution_time_wrapper import get_execution_time_print
from yaml import safe_load as load_yaml
from typing import Dict, Any
import numpy as np
from numpy import expand_dims, array
from keras.utils import load_img, img_to_array
from keras_vggface.utils import preprocess_input
from typing import List


@get_execution_time_print
def load_config(path: str = "repeat_src/run/config_dan_fcn.yml") -> Dict[str, Any]:
    with open(path, "r") as file:
        config_params = load_yaml(file)
    return config_params


def correct(train_y, pred_val_y):
    # FIXME: I have no idea what this method is correcting for
    train_std = np.std(train_y)
    val_std = np.std(pred_val_y)
    mean = np.mean(pred_val_y)
    pred_val_y = np.array(pred_val_y)
    pred_val_y = mean + (pred_val_y - mean) * train_std / val_std
    return pred_val_y


def load_images(file_list: List[str], batch_size: int) -> array:
    if batch_size > len(file_list):
        batch_size = len(file_list)

    # NOTE: the while true allow to iterate on a loop through the yield
    # values. Without it, only the first value would be yielded each time
    while True:
        count: int = 0
        x: list = []
        for path in file_list:
            x_temp = load_img(path)
            x_temp = img_to_array(x_temp)
            x_temp = expand_dims(x_temp, axis=0)
            x_temp = preprocess_input(x_temp, version=2)

            count += 1
            x.append(x_temp)
            if count % batch_size == 0 and count != 0:
                x = array(x)
                x = x.reshape(batch_size, 256, 256, 3)
                yield x
                x = []
