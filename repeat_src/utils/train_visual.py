from typing import List, Tuple

import numpy as np
from keras.models import Model
from numpy import array, ndarray
from pandas import read_csv
from tqdm import tqdm

from repeat_src.utils import correct
from repeat_src.utils.loss import ccc
from repeat_src.utils.visual import generator_test


def validate_epoch(
    model: Model,
    epoch: int,
    best_ccc: float,
    validation_file_list: ndarray,
    validation_label_list: ndarray,
    train_label_list: ndarray,
    batch_size: int,
    last_train_str: str,
) -> Tuple[Model, float]:
    prediction = model.predict_generator(
        generator_test(validation_file_list, batch_size),
        validation_file_list.shape[0] // batch_size + 1,
        verbose=0,
    )
    prediction_a = np.reshape(prediction[0], (1, -1))[0][
        0 : validation_file_list.shape[0]
    ]
    prediction_v = np.reshape(prediction[1], (1, -1))[0][
        0 : validation_file_list.shape[0]
    ]
    a_ccc, _ = ccc(validation_label_list[:, 0], prediction_a)
    a_ccc2, _ = ccc(
        validation_label_list[:, 0], correct(train_label_list[:, 0], prediction_a)
    )
    v_ccc, _ = ccc(validation_label_list[:, 1], prediction_v)
    v_ccc2, _ = ccc(
        validation_label_list[:, 1], correct(train_label_list[:, 1], prediction_v)
    )
    last_val_str = " [validate]-accc:%.4f(%.4f)-vccc:%.4f(%.4f)" % (
        a_ccc,
        a_ccc2,
        v_ccc,
        v_ccc2,
    )
    print(last_train_str + last_val_str, end="\n", flush=False)
    if a_ccc + v_ccc > best_ccc:
        best_ccc = v_ccc + a_ccc
        best_epoch = epoch
        print(
            f"Current best epoch: {best_epoch} with ccc: {best_ccc}, ccca: {a_ccc}, cccv: {v_ccc}"
        )
        model.save_weights("best_models.nosync/CNN_weights12_fin.h5")
    return model, best_ccc


def train_epoch(
    model: Model,
    epoch: int,
    number_of_epochs: int,
    number_of_batches: int,
    train_file_list: ndarray,
    train_label_list: ndarray,
    batch_size: int,
    current_batch_idx: int,
    max_length: int,
    last_train_str: str,
    sum_loss: float,
) -> Tuple[Model, str, float]:

    file_list = train_file_list[
        current_batch_idx * batch_size : (current_batch_idx + 1) * batch_size
    ]
    label_list = train_label_list[
        current_batch_idx * batch_size : (current_batch_idx + 1) * batch_size, :
    ]
    x, y1, y2 = [], [], []
    for j, path in enumerate(file_list):
        x_temp = read_csv(path).values

        if x_temp.shape[0] < max_length:
            x_temp = np.concatenate(
                (
                    x_temp,
                    np.zeros((max_length - x_temp.shape[0], x_temp.shape[1])),
                ),
                axis=0,
            )
        else:
            index_temp = np.arange(x_temp.shape[0] // 5) * 5
            rand_index = np.random.random(len(index_temp)) * 5
            index_temp = index_temp + rand_index.astype(int)
            if len(index_temp) >= max_length:
                index_temp = index_temp[0:max_length]
                x_temp = x_temp[index_temp, :]
            else:
                x_temp = x_temp[index_temp, :]
                x_temp = np.concatenate(
                    (
                        x_temp,
                        np.zeros((max_length - x_temp.shape[0], x_temp.shape[1])),
                    ),
                    axis=0,
                )
        x.append(x_temp)
        y1.append(label_list[j, 0])
        y2.append(label_list[j, 1])
    x = np.array(x)
    x = x.reshape(batch_size, max_length, -1).astype("float32")
    y1 = np.array(y1)
    y2 = np.array(y2)

    loss_value = model.train_on_batch(x, [y1, y2])
    sum_loss += loss_value[0]
    last_train_str = "\r[epoch:%d/%d, steps:%d/%d]-loss:%.4f" % (
        epoch + 1,
        number_of_epochs,
        current_batch_idx + 1,
        number_of_batches,
        sum_loss / (current_batch_idx + 1),
    )
    print(last_train_str, end="\n", flush=False)
    return model, last_train_str, sum_loss


def custom_train(
    model: Model,
    train_file_list: ndarray,
    train_label_list: array,
    validation_file_list: ndarray,
    validation_label_list: array,
    max_length: float,
    number_of_epochs: int = 1,
    batch_size: int = 1,
) -> Model:
    """This method train the custom CNN models on video frames, taken as
    features extracted from a VGG16-Face model.

    Parameters
    ----------
    model : Model
        model variable, to be trained
    train_file_list : ndarray
        list of file containing the extracted features (inside csv)
    number_of_epochs : int, optional
        number of epochs to train the models with, by default 1
    batch_size : int, optional
        batch size, by default 1

    Returns
    -------
    Model
        returns the trained model
    """

    number_of_batches = int(len(train_file_list) / batch_size)
    best_ccc = 0

    for epoch in range(number_of_epochs):
        # training step
        index = np.arange(len(train_file_list))
        np.random.shuffle(index)
        train_file_list = train_file_list[index]
        train_label_list = train_label_list[index, :]
        sum_loss = 0
        last_train_str = ""
        for i in tqdm(range(number_of_batches), desc=f"Training epoch [{epoch}]"):
            model, last_train_str, sum_loss = train_epoch(
                model=model,
                train_file_list=train_file_list,
                train_label_list=train_label_list,
                epoch=epoch,
                current_batch_idx=i,
                batch_size=batch_size,
                number_of_epochs=number_of_epochs,
                number_of_batches=number_of_batches,
                max_length=max_length,
                sum_loss=sum_loss,
                last_train_str=last_train_str,
            )
            model, best_ccc = validate_epoch(
                model=model,
                epoch=epoch,
                best_ccc=best_ccc,
                validation_file_list=validation_file_list,
                validation_label_list=validation_label_list,
                train_label_list=train_label_list,
                batch_size=batch_size,
                last_train_str=last_train_str,
            )

    return model
