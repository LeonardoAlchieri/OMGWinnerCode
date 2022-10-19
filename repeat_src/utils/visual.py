from pandas import read_csv
from numpy.random import seed as set_seed, shuffle as random_shuffle
from numpy import arange, concatenate, zeros, array, reshape, float32

from keras.models import Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace

from codes.CNN_model import model as customCNN


def get_model(model_name: str, model_params: dict) -> Model:

    if model_name == "cnn":
        return customCNN(**model_params)
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented")


def get_backbone_model(input_shape: tuple = (256, 256, 3)) -> Model:
    VGGFace_resnet50_model = VGGFace(
        model="vgg16", include_top=False, input_shape=(256, 256, 3), pooling="avg"
    )
    for layer in VGGFace_resnet50_model.layers:
        layer.trainable = False

    input_tensor = Input(input_shape)
    outputs = VGGFace_resnet50_model(input_tensor)
    model = Model(input_tensor, outputs, name="vgg16")
    return model


def generator_test(
    file_list, batch_size, shuffle=False, random_seed=None, max_length=64
):
    while True:
        if shuffle:
            if random_seed != None:
                random_seed += 1
                set_seed(random_seed)
            index = arange(file_list.shape[0])
            random_shuffle(index)
            file_list = file_list[index]
        count = 0
        x, y = [], []
        for i, path in enumerate(file_list):
            x_temp = read_csv(path)
            x_temp = x_temp.values
            if x_temp.shape[0] < max_length:
                x_temp = concatenate(
                    (x_temp, zeros((max_length - x_temp.shape[0], x_temp.shape[1]))),
                    axis=0,
                )
            else:
                index_temp = arange(x_temp.shape[0] // 5) * 5
                if len(index_temp) >= max_length:
                    index_temp = index_temp[0:max_length]
                    x_temp = x_temp[index_temp, :]
                else:
                    x_temp = x_temp[index_temp, :]
                    x_temp = concatenate(
                        (
                            x_temp,
                            zeros((max_length - x_temp.shape[0], x_temp.shape[1])),
                        ),
                        axis=0,
                    )
            count += 1
            x.append(x_temp)
            if count % batch_size == 0 and count != 0:
                x = array(x)
                x = x.reshape(batch_size, max_length, -1).astype("float32")
                yield x
                x = []
