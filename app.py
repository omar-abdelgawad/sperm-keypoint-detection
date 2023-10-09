import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import argparse
from ultralytics import YOLO
from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
from collections import defaultdict
from scipy.fft import rfft, rfftfreq
from itertools import cycle
from typing import Optional
from typing import Sequence
from typing import Any

# TODO: seriously try to use cv2.morphologyEx to remove noise it has great potential.
# TODO: activate cuda on this device and record the steps.
# TODO: GUI using tkinter
# TODO: refactor projection to be cleaner (maybe using sperm class).
# TODO: Tracking should be enhanced + why is it skipping ids??
# TODO: Add choices to magnification
# TODO: estimate the amplitude and the head frequency.
# TODO: Maybe try to interpolate the points in a polynomial instead of connecti ng them with a line.
# TODO: change model path to be absolute depending on the executable.
# TODO: remove warnings

# constants
MODEL_PATH = "./model/last.pt"
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
X_Y_ID_OFFSET = 10  # px
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 1
TEXT_COLOR = (0, 165, 255)
BBOX_COLOR = GREEN
THICKNESS = 2
POINT_RADIUS = 3
COLOR_LIST = cycle(
    [
        RED,
        GREEN,
        BLUE,
        (128, 128, 0),
        (0, 128, 128),
        (128, 0, 128),
        (255, 128, 0),
        (0, 128, 255),
        (128, 0, 255),
        (255, 0, 128),
    ]
)
OVERLAY_IMAGE_SAMPLE_RATE = 15
SPLINE_DEG = 2
NUM_POINTS_ON_FLAGELLUM = 100
POINTS_1_TO_4_COLOR = GREEN
POINTS_5_TO__COLOR = BLUE
PROJECTION_LINE_COLOR = BLUE
# directories
OUT_DIR = "out"
OUT_VIDEO_FOLDER = "videos"


class CustomDefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory:
            dict.__setitem__(self, key, self.default_factory(key))
            return self[key]
        else:
            defaultdict.__missing__(self, key)


@dataclass
class Sperm:
    """Sperm class for holding and displaying data related to a sperm in a video defined by an id."""

    id: int = field()
    sperm_overlay_image_shape: InitVar[tuple] = field()
    sperm_overlay_image: np.ndarray = field(default=np.zeros(sperm_overlay_image_shape))
    p_num_5: list[Any] = field(default_factory=list, repr=False)
    p_num_6: list[Any] = field(default_factory=list, repr=False)
    p_num_7: list[Any] = field(default_factory=list, repr=False)
    p_num_8: list[Any] = field(default_factory=list, repr=False)
    head_angle: list[Any] = field(default_factory=list, repr=False)
    sperm_image: Optional[np.ndarray] = field(default=None)

    def save_amplitude_figures(self, out_dir: str) -> None:
        """Creates and saves the amplitude figures of last 4 points of the Sperm.

        Args:
            out_dir(str): dir to image to.

        Returns:
            None"""
        title = f"Signed Amplitude of last 4 points for id {self.id}"
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
        axes[0, 0].plot(self.p_num_5)
        axes[0, 0].set_title("point 5")
        axes[0, 1].plot(self.p_num_6)
        axes[0, 1].set_title("point 6")
        axes[1, 0].plot(self.p_num_7)
        axes[1, 0].set_title("point 7")
        axes[1, 1].plot(self.p_num_8)
        axes[1, 1].set_title("point 8")
        plt.savefig(fname=f"{os.path.join(out_dir,title)}.jpeg")
        plt.close(fig)

    def save_head_frequency_figure(self, out_dir: str) -> None:
        """Creates and saves the head_frequency figure.

        Args:
            out_dir(str): dir to figure to.

        Returns:
            None"""
        title = f"head angle vs frame for id {self.id}"
        xlabel = "frame count"
        ylabel = "angle"
        fig, ax = plt.subplots()
        fig.suptitle(title)
        fig.text(0.5, 0.04, xlabel, ha="center")
        fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")
        ax.plot(self.head_angle)
        plt.savefig(fname=f"{os.path.join(out_dir,title)}.jpeg")
        plt.close(fig)

    def save_fft_graph_for_head_frequency(
        self, sampling_rate: int, out_dir: str
    ) -> None:
        """Creates and saves the fft graph for the head frequency and estimates it.

        Args:
            id_num(int): id of the sperm.
            signal(list): list of points of graph in time domain.
            sampling_rate(int): rate of sampling to determine actual frequencies.
            out_dir(str): dir to write figure to.

        Returns:
            (None)"""
        signal = self.head_angle
        n = len(signal)
        normalize = n / 2
        fourier: np.ndarray = np.array(rfft(signal))
        frequency_axis = rfftfreq(n, d=1.0 / sampling_rate)
        norm_amplitude = np.abs(fourier / normalize)
        estimated_frequency = estimate_freq(frequency_axis, norm_amplitude)

        title = f"fourier transform of head frequency for id {self.id}"
        xlabel = "frequencies"
        ylabel = "norm amplitude"
        fig, ax = plt.subplots()
        fig.suptitle(title)
        fig.text(0.5, 0.04, xlabel, ha="center")
        fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")
        ax.plot(frequency_axis, norm_amplitude)
        ax.axvline(
            x=estimated_frequency,
            color="red",
            linestyle="--",
            label=f"Estimated frequency = {estimated_frequency:.1f}",
        )
        ax.legend()
        # ax.set_xlim(0, 50)
        plt.savefig(fname=f"{os.path.join(out_dir,title)}.jpeg")
        plt.close(fig)


def find_fps(video_path: str) -> float:
    """Returns the fps of a video from its path.

    Args:
        video_path(str): relative or absolute path to video.

    Returns:
        (float): fps of the video if it exists."""
    return cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)


def vec_angle(vec_1: np.ndarray, vec_2: np.ndarray) -> float:
    """Returns the angle in degrees between two vectors."""
    vec_1 = vec_1.reshape(-1)
    vec_2 = vec_2.reshape(-1)
    rise = vec_2[1] - vec_1[1]
    run = vec_2[0] - vec_1[0]
    return np.degrees(np.arctan2(rise, run))


def write_video_from_img_array(
    img_array: list[np.ndarray], input_video_path: str
) -> None:
    """Write Video file to out/videos/projection_overlay+orig_video_name.

    Args:
        img_array(list[np.ndarray]): list of images that make the videos.
        input_video_path(str): path of input video.

    Returns:
        (None)"""
    orig_video_name = os.path.split(input_video_path)[1]
    height, width, _ = img_array[0].shape
    size = width, height
    overlay_video_name = "projection_overlay_" + orig_video_name
    video_path = os.path.join(OUT_DIR, OUT_VIDEO_FOLDER, overlay_video_name)
    out_vid = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"DIVX"), find_fps(input_video_path), size)  # type: ignore
    for img in img_array:
        out_vid.write(img)
    out_vid.release()


def estimate_freq(frequency_axis: np.ndarray, norm_amplitude: np.ndarray) -> float:
    """Estimates the frequency of the sperm's head by analyzing the graph in the frequency domain.

    Args:
        frequency_axis(np.ndarray): array of all possible frequencies/domain.
        norm_amplitude(np.ndarray): array of amplitudes for corresponding frequencies.

    Returns:
        (float): Estimated frequency of sperm's head"""
    is_increasing = norm_amplitude > np.roll(norm_amplitude, 1)
    is_decreasing = norm_amplitude > np.roll(norm_amplitude, -1)
    is_critical_point = is_increasing & is_decreasing
    is_critical_point[[0, -1]] = False
    norm_amplitude_tmp = np.array(norm_amplitude)
    norm_amplitude_tmp[~is_critical_point] = 0
    amp_index = np.argmax(norm_amplitude_tmp)
    return frequency_axis[amp_index]


def draw_head_ellipse(
    img: np.ndarray, v1: np.ndarray, v2: np.ndarray, color: tuple[int, int, int]
) -> None:
    center_coordinate = tuple((v1 + v2) // 2)
    dist = int(np.linalg.norm(v2 - v1))
    axes_length = (dist // 2 + 10, dist // 3)
    angle = vec_angle(v1, v2)
    start_angle = 0
    end_angle = 360
    cv2.ellipse(
        img,
        center_coordinate,
        axes_length,
        angle,
        start_angle,
        end_angle,
        color,
        THICKNESS,
    )


def draw_overlay_image(points, image: np.ndarray) -> None:
    """Takes list of keypoints and draws an ellipse using the first two then connects the rest.

    Args:
        points: list of keypoints.
        image(np.ndarray): image to draw on.

    Returns:
        (None)"""
    color = next(COLOR_LIST)
    points = np.array(points)
    draw_head_ellipse(image, points[0], points[1], color)
    points = points[1:].reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=False, color=color, thickness=THICKNESS)


def file_or_dir_exist(path: str) -> str:
    """raises a type exception if a path is not valid and returns it otherwise."""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"expected a valid path, got {path!r}")

    return path


def is_valid_magnification(mag: str) -> int:
    """Should determine if magnification is valid and return a number to use in calculations."""
    return 0
    raise NotImplementedError


def draw_bbox_and_id(
    image: np.ndarray,
    top_left_point: tuple[int, int],
    bottom_right_point: tuple[int, int],
    id: int,
) -> None:
    """Draws bounding box and write id on top left of bbox."""
    cv2.rectangle(image, top_left_point, bottom_right_point, BBOX_COLOR, THICKNESS)
    cv2.putText(
        image,
        f"{id}",
        (top_left_point[0] + X_Y_ID_OFFSET, top_left_point[1] + X_Y_ID_OFFSET),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        THICKNESS,
    )


def project_and_draw_points(
    image: np.ndarray, v1: np.ndarray, v2: np.ndarray, points: np.ndarray, track: dict
) -> list[np.ndarray]:
    straight_line_projection_points: list[np.ndarray] = [v1, v2]
    for i, p1 in enumerate(points, start=3):
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
        straight_line_projection_points.append(projection_pt)

        cv2.circle(
            image,
            v3,
            POINT_RADIUS,
            POINTS_1_TO_4_COLOR if i < 5 else POINTS_5_TO__COLOR,
            THICKNESS,
        )
        cv2.line(image, v3, projection_pt, GREEN, 2)
        if i > 4:
            track[f"p{i}"].append(projection_length)
    return straight_line_projection_points


def handle_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        default=os.path.join(".", "input_videos"),
        type=file_or_dir_exist,
        help="specify the input file path or dir of files. (defualt: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--magnif",
        required=True,
        type=is_valid_magnification,
        help="specify the microscope magnification used. No default value",
    )
    parser.add_argument(
        "-r",
        "--rate",
        required=True,
        type=int,
        help="specify the sampling rate of the input video(s) for calculating the fourier transform. No default value.",
    )
    args = parser.parse_args(argv)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    global OUT_DIR
    args = handle_parser(argv)
    input_video_path = args.input_path
    input_video_name = os.path.split(input_video_path)[1]
    OUT_DIR = os.path.join(OUT_DIR, os.path.splitext(input_video_name)[0])
    model = YOLO(MODEL_PATH)
    lstresults = model.track(
        source=input_video_path,
        save=True,
        show_conf=False,
        show_labels=True,
        project=OUT_DIR,
        name=OUT_VIDEO_FOLDER,
    )
    if model.device is None or model.device.type != "cuda":
        print(f"Can't find gpu/cuda. Used {model.device} instead.")
    else:
        print("Used cuda during inference.")

    track_history_dict: defaultdict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "p5": [],
            "p6": [],
            "p7": [],
            "p8": [],
            "head_angle": [],
            "sperm_image": None,
            "overlay_image": np.zeros_like(lstresults[0].orig_img),
        }
    )
    overlay_img_array: list[np.ndarray] = []
    for img_ind, result in enumerate(lstresults):
        img = np.array(result.orig_img)
        boxes: list = result.boxes.xyxy.int().cpu().tolist()
        keypoints: list = result.keypoints.xy.int().cpu().tolist()
        try:
            ids: list = result.boxes.id.int().cpu().tolist()
        except AttributeError:
            continue

        for obj_bbox_xyxy, track_id, obj_keypoints in zip(boxes, ids, keypoints):
            # bbox and id preparation
            x1, y1, x2, y2 = obj_bbox_xyxy
            track = track_history_dict[track_id]
            draw_bbox_and_id(img, (x1, y1), (x2, y2), track_id)

            # drawing on the overlay sperm image
            if img_ind % OVERLAY_IMAGE_SAMPLE_RATE == 0:
                draw_overlay_image(obj_keypoints, track["overlay_image"])

            v1 = np.array(obj_keypoints[0])
            v2 = np.array(obj_keypoints[1])

            track["head_angle"].append(vec_angle(v1, v2))
            if track["sperm_image"] is None:
                track["sperm_image"] = np.array(result.orig_img[y1:y2, x1:x2])

            straight_line_projection_points = project_and_draw_points(
                img, v1, v2, obj_keypoints[2:], track
            )
            # drawing straight line on overlay_video
            for pt1, pt2 in zip(
                straight_line_projection_points, straight_line_projection_points[1:]
            ):
                cv2.line(img, tuple(pt1), tuple(pt2), PROJECTION_LINE_COLOR, THICKNESS)
        overlay_img_array.append(img)
    # start writing files to the out directories
    # Create out Directory
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    print("Writing Overlayed video.")
    write_video_from_img_array(overlay_img_array, input_video_path)

    print("Writing ID folders")
    for id, track in track_history_dict.items():
        id_out_dir = os.path.join(OUT_DIR, f"id_{id}")
        if not os.path.exists(id_out_dir):
            os.makedirs(id_out_dir)
        cv2.imwrite(
            os.path.join(id_out_dir, f"id_{id}_sperm_image.jpeg"),
            track["sperm_image"],
        )
        save_amplitude_figures(id, track, id_out_dir)
        save_head_frequency_figure(id, track["head_angle"], id_out_dir)
        save_fft_graph_for_head_frequency(
            id, track["head_angle"], args.rate, id_out_dir
        )
        cv2.imwrite(
            os.path.join(id_out_dir, f"id_{id}_sperm_overlay_image.jpeg"),
            track["overlay_image"],
        )

    print("Task Finished succesfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
