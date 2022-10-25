from typing import Tuple

import numpy as np
from keras.models import Model
from numpy import ndarray

from repeat_src.utils import correct
from repeat_src.utils.loss import ccc


def custom_train(
    model: Model,
    train_data: ndarray,
    train_label_list: ndarray,
    validation_data: ndarray,
    validation_label_list: ndarray,
    number_of_epochs: int,
    batch_size: int,
) -> Tuple[float, int, Model]:

    best_ccc = 0
    best_epoch = 0

    n_batch = int(len(train_data) / batch_size)
    for epoch in range(number_of_epochs):
        # training step
        index = np.arange(train_data.shape[0])
        np.random.shuffle(index)
        train_data = train_data[index, :]
        train_label_list = train_label_list[index, :]
        sum_loss = 0
        last_train_str = ""
        for i in range(n_batch):
            x = train_data[i * batch_size : (i + 1) * batch_size, :]
            y1 = train_label_list[i * batch_size : (i + 1) * batch_size, 0]
            y2 = train_label_list[i * batch_size : (i + 1) * batch_size, 1]

            loss_value = model.train_on_batch(x, [y1, y2])
            sum_loss += loss_value[0]
            last_train_str = "\r[epoch:%d/%d, steps:%d/%d]-loss:%.4f" % (
                epoch + 1,
                number_of_epochs,
                i + 1,
                n_batch,
                sum_loss / (i + 1),
            )
            print(last_train_str, end="      ", flush=True)

        # validating for mse & ccc
        prediction = model.predict(validation_data)
        prediction_a = np.reshape(prediction[0], (1, -1))[0]
        prediction_v = np.reshape(prediction[1], (1, -1))[0]
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
        print(last_train_str + last_val_str, end="      ", flush=True)
        if epoch >= 99 and (a_ccc + v_ccc) / 2.0 > best_ccc:
            best_ccc = (a_ccc + v_ccc) / 2.0
            best_epoch = epoch + 1
            model.save_weights("./best_models.nosync/AudCNN_weights_fin.h5")

    return best_ccc, best_epoch, model
