import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from ultralytics import YOLO
from collections import defaultdict
from scipy.fft import rfft, rfftfreq
from typing import Optional
from typing import Literal

# TODO: Maybe try to interpolate the points in a polynomial instead of connecting them with a line.

# directories
OUT_DIR = "./out"
OUT_VIDEO_DIR = "videos"

# inputs
INPUT_VIDEO_PATH = "other_data/sperm_vids/good_quality/f7 737.052.avi"
MAGNIFICATION = None
FPS = None

# constants
MODEL_PATH = "./model/last.pt"
VIDEO_NAME = os.path.split(INPUT_VIDEO_PATH)[1]
X_Y_ID_OFFSET: Literal[10] = 10  # px
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 1
COLOR = (0, 165, 255)
THICKNESS = 2


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


def write_video_from_img_array(img_array: list[np.ndarray], out_path) -> None:
    height, width, layers = img_array[0].shape
    size = width, height
    out_vid = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"DIVX"), FPS, size)  # type: ignore
    for img in img_array:
        out_vid.write(img)
    out_vid.release()


def find_signed_projection_length(
    projection_point, orig_point, projection_line
) -> float:
    projection_length = np.linalg.norm(projection_point - orig_point) * np.sign(
        np.cross(np.squeeze(projection_line), np.squeeze(b))
    )
    return float(projection_length)


def save_amplitude_figures(id_num, track_history_dict) -> None:
    title = f"Signed Amplitude of last 4 points for id:{id_num}"
    xlabel = "frame count"
    ylabel = "distance between point and head axis in pixels"
    fig, axes = plt.subplots(2, 2, sharey=True, sharex=True)
    fig.suptitle(title)
    fig.text(0.5, 0.04, xlabel, ha="center")
    fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")
    for i in range(2):
        for j in range(2):
            axes[i, j].axhline(
                y=0, color="black", linestyle="-", linewidth=2, label=None
            )
    axes[0, 0].plot(track_history_dict[id_num]["p4"])
    axes[0, 0].set_title("point 4")
    axes[0, 1].plot(track_history_dict[id_num]["p5"])
    axes[0, 1].set_title("point 5")
    axes[1, 0].plot(track_history_dict[id_num]["p6"])
    axes[1, 0].set_title("point 6")
    axes[1, 1].plot(track_history_dict[id_num]["p7"])
    axes[1, 1].set_title("point 7")
    plt.savefig(fname=f"{os.path.join(OUT_DIR,title)}.jpeg")


def save_head_frequency_figure():
    pass


def main(argv: Optional[list[str]] = None):
    model = initialize_model(MODEL_PATH)
    lstresults = model.track(
        source=INPUT_VIDEO_PATH,
        save=True,
        show_conf=False,
        show_labels=True,
        project=OUT_DIR,
        name=OUT_VIDEO_DIR,
    )

    track_history_dict = defaultdict(
        lambda: {"p5": [], "p6": [], "p7": [], "p8": [], "head_angle": []}
    )
    overlay_img_array: list[np.ndarray] = []
    for result in lstresults:
        img = np.array(result.orig_img)
        boxes = result.boxes.xyxy.int().cpu().tolist()
        ids = result.boxes.id.int().cpu().tolist()
        keypoints = result.keypoints.xy.int().cpu().tolist()

        for obj_bbox_xyxy, track_id, obj_keypoints in zip(boxes, ids, keypoints):
            # bbox and id preparation
            x1, y1, x2, y2 = obj_bbox_xyxy
            track = track_history_dict[track_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(
                img,
                f"{track_id}",
                (x1 + X_Y_ID_OFFSET, y1 + X_Y_ID_OFFSET),
                FONT,
                FONT_SCALE,
                COLOR,
                THICKNESS,
            )
            v1 = np.array(obj_keypoints[0])
            v2 = np.array(obj_keypoints[1])
            track["head_angle"].append(vec_angle(v1, v2))
            line_to_draw: list[np.ndarray] = [v1, v2]
            for i, p1 in enumerate(obj_keypoints[2:], start=2):
                v3 = np.array(p1)

                reshape_vec_2d = lambda arr: arr.reshape(len(arr), 1)
                v1, v2, v3 = map(reshape_vec_2d, (v1, v2, v3))
                projection_line = v2 - v1
                b = v3 - v1
                projection_pt = v1 + np.dot(
                    (
                        (np.dot(projection_line, projection_line.T))
                        / (np.dot(projection_line.T, projection_line) + 1e-5)
                    ),
                    b,
                )
                projection_pt = projection_pt.astype(np.int32)

                v3 = v3.reshape(-1)
                projection_pt = projection_pt.reshape(-1)
                projection_length = np.linalg.norm(projection_pt - v3) * np.sign(
                    np.cross(np.squeeze(projection_line), np.squeeze(b))
                )
                line_to_draw.append(projection_pt)

                cv2.circle(img, v3, 3, (0, 0, 200) if i < 3 else (255, 0, 0), 4)
                cv2.line(img, v3, projection_pt, (0, 255, 0), 2)
                if i > 3:
                    track[f"p{i+1}"].append(projection_length)
            for pt1, pt2 in zip(line_to_draw, line_to_draw[1:]):
                cv2.line(img, tuple(pt1), tuple(pt2), (255, 0, 0), 4)
        overlay_img_array.append(img)
    # start writing files to the out directories
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    write_video_from_img_array(overlay_img_array, os.path.join(OUT_DIR, VIDEO_NAME))
    # Create out Directory


if __name__ == "__main__":
    main()
