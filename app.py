import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from ultralytics import YOLO
from collections import defaultdict
from scipy.fft import rfft, rfftfreq
from typing import Optional
from typing import Literal
from typing import Any

# TODO: produce an image of an overlay of the sperm across multiple frames.
# TODO: Maybe try to interpolate the points in a polynomial instead of connecting them with a line.
# TODO: estimate the amplitude and the head frequency.
# TODO: GUI using tkinter?????

# inputs
INPUT_VIDEO_PATH = "other_data/sperm_vids/good_quality/f5 736.avi"
MAGNIFICATION = None
SAMPLING_RATE = 736

# constants
MODEL_PATH = "./model/last.pt"
VIDEO_NAME = os.path.split(INPUT_VIDEO_PATH)[1]
X_Y_ID_OFFSET: Literal[10] = 10  # px
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 1
COLOR = (0, 165, 255)
THICKNESS = 2

# directories
OUT_DIR = os.path.join("./out", os.path.splitext(VIDEO_NAME)[0])
OUT_VIDEO_FOLDER = "videos"


def find_fps(video_path):
    return cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)


def vec_angle(vec_1, vec_2) -> float:
    rise = vec_2[1] - vec_1[1]
    run = vec_2[0] - vec_1[0]
    return np.degrees(np.arctan2(rise, run))


def write_video_from_img_array(img_array: list[np.ndarray], orig_video_name) -> None:
    height, width, _ = img_array[0].shape
    size = width, height
    overlay_video_name = "projection_overlay_" + orig_video_name
    video_path = os.path.join(OUT_DIR, OUT_VIDEO_FOLDER, overlay_video_name)
    out_vid = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"DIVX"), find_fps(INPUT_VIDEO_PATH), size)  # type: ignore
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


def save_amplitude_figures(id_num: int, id_dict: dict, out_dir: str) -> None:
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
    axes[0, 0].plot(id_dict["p5"])
    axes[0, 0].set_title("point 5")
    axes[0, 1].plot(id_dict["p6"])
    axes[0, 1].set_title("point 6")
    axes[1, 0].plot(id_dict["p7"])
    axes[1, 0].set_title("point 7")
    axes[1, 1].plot(id_dict["p8"])
    axes[1, 1].set_title("point 8")
    plt.savefig(fname=f"{os.path.join(out_dir,title)}.jpeg")
    plt.close(fig)


def save_head_frequency_figure(id_num: int, id_dict: dict, out_dir: str):
    title = f"head angle vs frame for id: {id_num}"
    xlabel = "frame count"
    ylabel = "angle"
    fig, ax = plt.subplots()
    fig.suptitle(title)
    fig.text(0.5, 0.04, xlabel, ha="center")
    fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")
    ax.plot(id_dict["head_angle"])
    plt.savefig(fname=f"{os.path.join(out_dir,title)}.jpeg")
    plt.close(fig)


def save_fft_graph_for_head_frequency(
    id_num: int, id_dict: dict, sampling_rate: int, out_dir: str
):
    signal = id_dict["head_angle"]
    n = len(signal)
    normalize = n / 2
    fourier: np.ndarray = np.array(rfft(signal))
    frequency_axis = rfftfreq(n, d=1.0 / sampling_rate)
    norm_amplitude = np.abs(fourier / normalize)
    estimated_frequency = estimate_freq(frequency_axis, norm_amplitude)

    title = f"fourier transform of head frequency for id: {id_num}"
    xlabel = "frequencies"
    ylabel = "norm amplitude"
    fig, ax = plt.subplots()
    fig.suptitle(title)
    fig.text(0.5, 0.04, xlabel, ha="center")
    fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")
    ax.plot(frequency_axis, norm_amplitude)
    ax.axvline(
        x=estimated_frequency, color="red", linestyle="--", label="Estimated frequency"
    )
    # ax.set_xlim(0, 50)
    plt.savefig(fname=f"{os.path.join(out_dir,title)}.jpeg")
    plt.close(fig)


def estimate_freq(frequency_axis: np.ndarray, norm_amplitude: np.ndarray):
    is_increasing = norm_amplitude > np.roll(norm_amplitude, 1)
    is_decreasing = norm_amplitude > np.roll(norm_amplitude, -1)
    is_critical_point = is_increasing & is_decreasing
    is_critical_point[[0, -1]] = False
    norm_amplitude_tmp = np.array(norm_amplitude)
    norm_amplitude_tmp[~is_critical_point] = 0
    amp_index = np.argmax(norm_amplitude_tmp)
    return frequency_axis[amp_index]


def main(argv: Optional[list[str]] = None):
    model = YOLO(MODEL_PATH)
    lstresults = model.track(
        source=INPUT_VIDEO_PATH,
        save=True,
        show_conf=False,
        show_labels=True,
        project=OUT_DIR,
        name=OUT_VIDEO_FOLDER,
    )
    if model.device is None or model.device.type != "cuda":
        print(f"Can't find gpu/cuda. Using {model.device} instead.")
    else:
        print("Used cuda for inference.")

    track_history_dict: defaultdict[
        int, dict[str, list[Any] | np.ndarray | None]
    ] = defaultdict(
        lambda: {
            "p5": [],
            "p6": [],
            "p7": [],
            "p8": [],
            "head_angle": [],
            "sperm_image": None,
            "overlay_image": [],
        }
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
            if track["sperm_image"] is None:
                track["sperm_image"] = np.array(result.orig_img[y1:y2, x1:x2])
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
                projection_length: float = np.linalg.norm(projection_pt - v3) * np.sign(
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
    # Create out Directory
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    write_video_from_img_array(overlay_img_array, VIDEO_NAME)
    for id in track_history_dict:
        id_out_dir = os.path.join(OUT_DIR, f"{id}")
        if not os.path.exists(id_out_dir):
            os.makedirs(id_out_dir)
        cv2.imwrite(
            os.path.join(id_out_dir, f"id:{id}_sperm_image.jpeg"),
            track_history_dict[id]["sperm_image"],
        )
        save_amplitude_figures(id, track_history_dict[id], id_out_dir)
        save_head_frequency_figure(id, track_history_dict[id], id_out_dir)
        save_fft_graph_for_head_frequency(
            id, track_history_dict[id], SAMPLING_RATE, id_out_dir
        )


if __name__ == "__main__":
    main()
