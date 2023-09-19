import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from ultralytics import YOLO
from collections import defaultdict
from typing import Optional

# directories
OUT_DIR = "./out"
OUT_VIDEO_DIR = "videos"

# constants
MODEL_PATH = "./model/last.pt"

# inputs
VIDEO_PATH = "other_data/sperm_vids/good_quality/f7 737.052.avi"
MAGNIFICATION = None
FPS = None


def vec_angle(vec_1, vec_2) -> float:
    rise = vec_2[1] - vec_1[1]
    run = vec_2[0] - vec_1[0]
    return np.degrees(np.arctan2(rise, run))


def initialize_model(model_path: str) -> YOLO:
    """initializes model and prints warning if device is not cuda.

    Args:
        model_path(str): path to model weights file.

    Returns:
        (YOLO): YOLO pose estimation model.
    """
    model = YOLO(MODEL_PATH)

    if model.device is None or model.device.type != "cuda":
        print(f"Can't find gpu/cuda. Using {model.device} instead.")

    return model

def write_video_from_img_array(img_array)
def main(argv: Optional[list[str]] = None):
    model = initialize_model(MODEL_PATH)
    lstresults = model.track(
        source=VIDEO_PATH,
        save=True,
        show_conf=False,
        show_labels=True,
        project=OUT_DIR,
        name=OUT_VIDEO_DIR,
    )
    track_history_dict = defaultdict(lambda: {'p1':[],'p4':[],'p5':[],'p6':[],'p7':[],'head_angle':[]})

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)


if __name__ == "__main__":
    main()
