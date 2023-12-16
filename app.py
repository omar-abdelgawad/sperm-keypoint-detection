"""Script for Main app entry point"""
import os
import sys
import argparse
import multiprocessing
from typing import Optional
from typing import Sequence

import numpy as np
import cv2
from ultralytics import YOLO

from src.sperm import Sperm
from src.gui import GUI
from src import cfg

# mandatory TODO(s):
# TODO: add all utility functions to utils.py
# TODO: add a progress bar
# TODO: add a logger
# TODO: add a button to open the output folder
# TODO: look up if not joining the thread(waiting for it) or making it daemon(not waiting at all for it) is a good idea.
# TODO: activate cuda on this device and record the steps.
# TODO: Make and deploy Github pages.
# TODO: replace all path handling to use pathlib
# Other TODO(s):
# TODO: seriously try to use cv2.morphologyEx to remove noise it has great potential.
# TODO: Tracking should be enhanced + why is it skipping ids??
# TODO: estimate the amplitude and the head frequency as single numbers in the end.
# TODO: Maybe try to interpolate the points in a polynomial instead of connecting them with a line.
# TODO: remove warnings from exe file
# TODO: add pytest

# constants
EXE_DIR = os.path.dirname(sys.argv[0])
MODEL_PATH = os.path.join(EXE_DIR, "model", "last.pt")

# directories
OUT_DIR = "out"
OUT_VIDEO_FOLDER = "videos"


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
        cfg.THICKNESS,
    )


def draw_overlay_image(points, image: np.ndarray) -> None:
    """Takes list of keypoints and draws an ellipse using the first two then connects the rest.

    Args:
        points: list of keypoints.
        image(np.ndarray): image to draw on.

    Returns:
        (None)"""
    color = next(cfg.COLOR_LIST)
    points = np.array(points)
    draw_head_ellipse(image, points[0], points[1], color)
    points = points[1:].reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=False, color=color, thickness=cfg.THICKNESS)


def file_or_dir_exist(path: str) -> str:
    """raises a type exception if a path is not valid and returns it otherwise."""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"expected a valid path, got {path!r}")

    return path


def is_valid_magnification(mag: str) -> str:
    """Should determine if magnification is valid and return a number to use in calculations."""
    if not mag in cfg.MAGNIFICATION_LIST:
        raise argparse.ArgumentTypeError(f"expected a valid magnification, got {mag!r}")
    return mag


def draw_bbox_and_id(
    image: np.ndarray,
    top_left_point: tuple[int, int],
    bottom_right_point: tuple[int, int],
    sperm_id: int,
) -> None:
    """Draws bounding box and write sperm_id on top left of bbox."""
    cv2.rectangle(
        image, top_left_point, bottom_right_point, cfg.BBOX_COLOR, cfg.THICKNESS
    )
    cv2.putText(
        image,
        f"{sperm_id}",
        (top_left_point[0] + cfg.X_Y_ID_OFFSET, top_left_point[1] + cfg.X_Y_ID_OFFSET),
        cfg.FONT,
        cfg.FONT_SCALE,
        cfg.TEXT_COLOR,
        cfg.THICKNESS,
    )


def project_and_draw_points(
    image: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    points: np.ndarray,
    sperm: Sperm,
    mag: str,
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
        projection_length: float = (
            np.linalg.norm(projection_pt - v3)
            * np.sign(np.cross(np.squeeze(projection_line), np.squeeze(b)))
            * cfg.PIXEL_SIZE_FOR_MAGNIFICATION[mag]
        )
        straight_line_projection_points.append(projection_pt)

        cv2.circle(
            image,
            v3,
            cfg.POINT_RADIUS,
            cfg.POINTS_1_TO_4_COLOR if i < 5 else cfg.POINTS_5_TO__COLOR,
            cfg.THICKNESS,
        )
        cv2.line(image, v3, projection_pt, cfg.GREEN, 2)
        if i == 5:
            sperm.p_num_5.append(projection_length)
        if i == 6:
            sperm.p_num_6.append(projection_length)
        if i == 7:
            sperm.p_num_7.append(projection_length)
        if i == 8:
            sperm.p_num_8.append(projection_length)

    return straight_line_projection_points


def handle_parser(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Handles the command line arguments.

    Args:
        argv (Optional[Sequence[str]]): 3 arguments are expected. Input path, magnification, sampling rate.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
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


def main() -> int:
    """Graphical user interface is here"""
    app = GUI(analyze_video)
    app.run()
    return 0


def analyze_video(argv: Optional[Sequence[str]]) -> int:
    """Main calculations and inference is done here"""
    global OUT_DIR
    args = handle_parser(argv)
    input_video_path = args.input_path
    input_video_name = os.path.split(input_video_path)[1]
    OUT_DIR = os.path.join(EXE_DIR, OUT_DIR, os.path.splitext(input_video_name)[0])

    print(MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print("finished loading model")
    lstresults = model.track(
        source=input_video_path,
        save=True,
        show_conf=False,
        show_labels=True,
        project=OUT_DIR,
        name=OUT_VIDEO_FOLDER,
    )
    if model.device is None or model.device.type != "cuda":
        print(f"Couldn't find gpu/cuda. Used {model.device} instead.")
    else:
        print("Used cuda during inference.")

    track_history_dict: dict[int, Sperm] = {}
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
            # get cur_sperm from track_history_dict and if it doesnt exist set default value
            if track_id not in track_history_dict:
                track_history_dict[track_id] = Sperm(
                    id=track_id,
                    sperm_overlay_image_shape=lstresults[0].orig_img.shape,
                )
            cur_sperm = track_history_dict[track_id]
            # drawing on the original video
            draw_bbox_and_id(img, (x1, y1), (x2, y2), track_id)

            # drawing on the overlay sperm image
            if img_ind % cfg.OVERLAY_IMAGE_SAMPLE_RATE == 0:
                draw_overlay_image(obj_keypoints, cur_sperm.sperm_overlay_image)

            v1 = np.array(obj_keypoints[0])
            v2 = np.array(obj_keypoints[1])

            cur_sperm.head_angle.append(vec_angle(v1, v2))
            if cur_sperm.sperm_image is None:
                cur_sperm.sperm_image = np.array(result.orig_img[y1:y2, x1:x2])

            straight_line_projection_points = project_and_draw_points(
                img, v1, v2, obj_keypoints[2:], cur_sperm, args.magnif
            )
            # drawing straight line on overlay_video
            for pt1, pt2 in zip(
                straight_line_projection_points, straight_line_projection_points[1:]
            ):
                cv2.line(
                    img,
                    tuple(pt1),
                    tuple(pt2),
                    cfg.PROJECTION_LINE_COLOR,
                    cfg.THICKNESS,
                )
        overlay_img_array.append(img)
    # start writing files to the out directories
    # Create out Directory
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    print("Writing Overlayed video.")
    write_video_from_img_array(overlay_img_array, input_video_path)

    print("Writing ID folders")
    for sperm_id, sperm in track_history_dict.items():
        sperm_id_out_dir = os.path.join(OUT_DIR, f"sperm_id_{sperm_id}")
        if not os.path.exists(sperm_id_out_dir):
            os.makedirs(sperm_id_out_dir)
        sperm.save_all_features(sperm_id_out_dir, args.rate)

    print("Task Finished succesfully.")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("Starting app.py")
    sys.exit(main())
