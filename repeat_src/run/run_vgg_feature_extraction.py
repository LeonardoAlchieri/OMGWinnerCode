from os.path import basename, join as join_paths
from numpy.random import seed as set_seed
from pandas import read_csv, DataFrame
from typing import Dict
from tqdm import tqdm
from glob import glob
from pathlib import Path

from tensorflow.keras.optimizers import SGD
from sys import path
path.append(".")
from repeat_src.utils import load_config, load_images
from repeat_src.utils.models import get_backbone_model


_filename: str = basename(__file__).split(".")[0][4:]


def get_paths(ground_truth_path: str, frames_path: str) -> Dict[str, str]:
    # make a list of unique videos (join video and utterance from the table with ground truths)
    ground_truth_df = read_csv(ground_truth_path)

    video_id_list: list = ground_truth_df.apply(
        lambda x: f"{x['video']}/{x['utterance'].split('.')[0]}", axis=1
    ).tolist()

    paths_to_faces: Dict[str] = {
        video_id: join_paths(frames_path, video_id) for video_id in video_id_list
    }
    return paths_to_faces


def main(random_state: int):
    set_seed(random_state)

    path_to_config: str = f"repeat_src/run/config_{_filename}.yml"

    print("Starting model training")
    configs = load_config(path=path_to_config)
    print("Configs loaded")

    model = get_backbone_model(input_shape=configs["input_shape"])

    paths_to_faces = get_paths(
        ground_truth_path=configs["ground_truth_path"],
        frames_path=configs["frames_path"],
    )

    count = 0
    for video_id, current_video_path in tqdm(
        paths_to_faces.items(), desc="Extracting features w/ VGG16-Face"
    ):
        video_frames_paths = glob(join_paths(current_video_path, "*.png"))

        current_video_features = model.predict_generator(
            load_images(video_frames_paths, configs["batch_size"]),
            (len(video_frames_paths) // configs["batch_size"]) + 1,
            verbose=1,
        )
        # NOTE: this saving sucks and takes a lot of space, but it is how
        # the origina authors did it
        current_video_features = current_video_features[0 : len(paths_to_faces), :]
        current_video_features = DataFrame(current_video_features)
        # FIXME: add folder creation/checking
        output_folder: str = join_paths(configs["saving_path"], f"{video_id.split('/')[0]}")
        
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        current_video_features.to_csv(
            join_paths(configs["saving_path"], f"{video_id}.csv"), index=None
        )
        del current_video_features
        count += 1
        if count == 4:
            print('Debugger stopper')
            break


if __name__ == "__main__":
    main(random_state=42)
