from glob import glob
from os.path import join as join_paths
from typing import List

from execution_time_wrapper import get_execution_time_print
from keras.models import Model
from numpy import array, concatenate, ndarray, zeros
from pandas import DataFrame, read_csv
from scipy.io.wavfile import read as read_audio

from codes.aud_CNN_model import model as aud_model


def get_model(model_name: str, model_params: dict) -> Model:

    if model_name == "cnn":
        return aud_model(**model_params)
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented")


def load_extract_opensmile_features(path_to_data: str) -> ndarray:
    """
    Load the extracted features from the opensmile library
    """
    data: DataFrame = read_csv(path_to_data, index_col=0)
    return data.values


@get_execution_time_print
def load_prepare_raw_audio(path_to_data: str, max_length: int) -> ndarray:
    """
    Load the raw audio files and prepare them for the model
    """
    audio_files_list: List[str] = glob(join_paths(path_to_data, "*/*.wav"))

    audio_arrays: List[ndarray] = []
    for audio_file in audio_files_list:
        current_audio_array: ndarray = read_audio(audio_file)[1]
        if len(current_audio_array) >= max_length:
            current_audio_array = current_audio_array[0:max_length]
        else:
            current_audio_array = concatenate(
                (current_audio_array, zeros(((max_length - len(current_audio_array)),)))
            )
        audio_arrays.append(current_audio_array)
    return array(audio_arrays)
