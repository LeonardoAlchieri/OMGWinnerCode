from glob import glob
from os import environ
from os.path import basename
from sys import path

from numpy import array, ndarray
from numpy.random import seed as set_seed
from pandas import read_csv

environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from warnings import filterwarnings

filterwarnings(
    "ignore",
    message="`Model.predict_generator` is deprecated and will be removed in a future version. Please use `Model.predict`, which supports generators.",
)
filterwarnings(
    "ignore", message="The `lr` argument is deprecated, use `learning_rate` instead."
)


from keras.models import Model
from tensorflow.config import set_visible_devices
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import set_random_seed as set_keras_seed

# FIXME: move to package imports, and not relative imports w/ path.append
path.append(".")
from repeat_src.utils import load_config
from repeat_src.utils.audio import get_model
from repeat_src.utils.train_visual import custom_train

_filename: str = basename(__file__).split(".")[0][4:]


def main(random_state: int):
    set_seed(random_state)
    set_keras_seed(random_state)

    path_to_config: str = f"repeat_src/run/config_{_filename}.yml"

    print("Starting model training")
    configs = load_config(path=path_to_config)
    print("Configs loaded")

    if configs["nogpu"]:
        set_visible_devices([], "GPU")

    model = get_model(
        model_name=configs["model_name"], model_params=configs["model_params"]
    )

    optimizer = SGD(**configs["optimizer_params"])

    model.compile(loss="mae", optimizer=optimizer)

    train_data = load_extract_opensmile_features(configs["audio_files"]["train_path"])
    validation_data = load_extract_opensmile_features(
        configs["audio_files"]["validation_path"]
    )
