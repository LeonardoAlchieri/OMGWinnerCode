from numpy import ndarray

from keras.models import Model

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