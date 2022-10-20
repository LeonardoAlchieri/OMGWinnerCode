from glob import glob
from os.path import basename
from os.path import join as join_paths
from typing import List

from execution_time_wrapper import get_execution_time_print
from numpy import arange, array, ndarray, save
from numpy.random import seed as set_seed
from opensmile import FeatureLevel, FeatureSet, Smile
from pandas import DataFrame, Series, read_csv
from xgboost import XGBClassifier

from repeat_src.utils import load_config

_filename: str = basename(__file__).split(".")[0][4:]

def clean_index(current_idx: tuple) -> list:
    new_idx = current_idx[0].split("/")[-2:]
    new_idx[-1] = new_idx[-1].split(".")[0]
    return "/".join(new_idx)

@get_execution_time_print
def extract_features(audio_files: List[str]) -> DataFrame:
    # loading & extracting features
    print("Loading & extracting features...")
    smile = Smile(
        feature_set=FeatureSet.ComParE_2016, feature_level=FeatureLevel.Functionals,
    )
    audio_features: DataFrame = smile.process_files(audio_files)
    audio_features.index = audio_features.index.map(clean_index)
    return audio_features

@get_execution_time_print
def load_labels(labels_path: str) -> Series:
    # loading labels
    print("Loading labels...")
    labels: DataFrame = read_csv(labels_path)
    labels.index = labels.apply(
        lambda x: f"{x['video']}/{x['utterance'].split('.')[0]}", axis=1
    )
    return labels

@get_execution_time_print
def main(random_state: int):

    set_seed(random_state)

    path_to_config: str = f"repeat_src/run/config_{_filename}.yml"

    print("Starting model training")
    configs = load_config(path=path_to_config)
    print("Configs loaded")
    
    audio_train_file: List[str] = glob(join_paths(configs['audio_train_file'], "*/*.wav"))
    audio_validation_file: List[str] = glob(join_paths(configs['audio_validation_file'], "*/*.wav"))
    train_labels_path: str = configs['train_labels_path']

    audio_train_features = extract_features(audio_train_file)
    audio_validation_features = extract_features(audio_validation_file)

    train_labels = load_labels(train_labels_path)

    # merging according to index
    print("Merging...")
    audio_train_features = audio_train_features.sort_index(inplace=False)
    train_labels = train_labels.sort_index(inplace=False)

    audio_train_features_with_labels = audio_train_features.merge(
        train_labels["EmotionMaxVote"], left_index=True, right_index=True
    )
    # extracting features and labels
    print("Extracting features and labels...")
    x_train: ndarray = audio_train_features_with_labels.values[:, :-1]
    y_train: ndarray = audio_train_features_with_labels.values[:, -1]

    # training xgboost model to identify emotions
    print("Training xgboost model...")
    model = XGBClassifier()
    model.fit(x_train, y_train)

    # get most importance 256 features
    print("Getting most importance 256 features...")
    feature_importante_idx = arange(len(model.feature_importances_))
    most_important_features_idx = array(
        Series(model.feature_importances_, index=feature_importante_idx)
        .sort_values(ascending=False)[:256]
        .index
    )
    # save the most important features to csv
    print("Saving the most important train features to csv...")
    x_train_most_important_features = x_train[:, most_important_features_idx]
    DataFrame(
        x_train_most_important_features, index=audio_train_features_with_labels.index
    ).to_csv("./features_extracted_audio/train_features.csv")

    print("Saving the most important validation features to csv...")
    x_validation_most_important_features = audio_validation_features.values[
        :, most_important_features_idx
    ]
    DataFrame(
        x_validation_most_important_features, index=audio_validation_features.index
    ).to_csv("./features_extracted_audio/validation_features.csv")

    # save the index of the features, as extracted from opensmile
    save("./features_extracted_audio/importante_idx.npy", most_important_features_idx)


if __name__ == "__main__":
    main(random_state=42)