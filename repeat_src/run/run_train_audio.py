from os import environ
from os.path import basename
from sys import path

from execution_time_wrapper import get_execution_time_print
from numpy import ndarray
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

try:
    from tensorflow.keras.utils import set_random_seed as set_keras_seed
except:
    print(f"Using lower tensorflow version")
    from tensorflow.random import set_seed as set_keras_seed

# FIXME: move to package imports, and not relative imports w/ path.append
path.append(".")
from repeat_src.utils import load_config
from repeat_src.utils.audio import (
    get_model,
    load_extract_opensmile_features,
    load_prepare_raw_audio,
)
from repeat_src.utils.train_audio import custom_train

_filename: str = basename(__file__).split(".")[0][4:]


@get_execution_time_print
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

    if configs["audio_files"]["use_raw_signal"]:
        train_data = load_prepare_raw_audio(
            configs["audio_files"]["train_path"],
            max_length=configs["model_params"]["timesteps"],
        )
        validation_data = load_prepare_raw_audio(
            configs["audio_files"]["validation_path"],
            max_length=configs["model_params"]["timesteps"],
        )
        if len(train_data) == 0 or len(validation_data) == 0:
            raise ValueError(
                "No data found for training. Please check if the paths are correct."
            )
    else:
        train_data = load_extract_opensmile_features(
            configs["audio_files"]["train_path"]
        )
        validation_data = load_extract_opensmile_features(
            configs["audio_files"]["validation_path"]
        )

    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")

    # load files with the ground truth for valence and arousal
    train_label_list: ndarray = read_csv(configs["ground_truth"]["train_path"])[
        ["valence", "arousal"]
    ].values
    validation_label_list: ndarray = read_csv(
        configs["ground_truth"]["validation_path"]
    )[["valence", "arousal"]].values

    best_ccc, best_epoch, model = custom_train(
        model=model,
        train_data=train_data,
        train_label_list=train_label_list,
        validation_data=validation_data,
        validation_label_list=validation_label_list,
        number_of_epochs=configs["number_of_epochs"],
        batch_size=configs["batch_size"],
    )
    print(f"Best CCC: {best_ccc} at epoch {best_epoch}")


if __name__ == "__main__":
    main(random_state=42)
