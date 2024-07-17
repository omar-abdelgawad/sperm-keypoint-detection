"""Counts the number of frames in a directory of videos."""
from pathlib import Path
import argparse
import cv2


def main():
    args = handle_parser()
    video_dir: Path = args.video_dir
    if not video_dir.exists():
        raise FileNotFoundError(f"{video_dir} does not exist.")
    if not video_dir.is_dir():
        raise NotADirectoryError(f"{video_dir} is not a directory.")
    total_num_frames = get_total_num_frames_recursive(video_dir)
    print(f"Total number of frames: {total_num_frames}")


def get_total_num_frames_recursive(video_dir: Path) -> int:
    """Returns the total number of frames in a directory hierarchy of vidoes and subfolders

    Args:
        video_dir (Path): path to folder containing videos and subfolders

    Returns:
        int: total number of frames in videos
    """
    total_num_frames = 0
    for video_path in video_dir.iterdir():
        if video_path.is_dir():
            total_num_frames += get_total_num_frames_recursive(video_path)
            continue
        if video_path.suffix not in [".mp4", ".avi"]:
            continue
        num_frames = get_frames_in_a_video(video_path)
        print(f"{video_path.name} has {num_frames} frames.")
        total_num_frames += num_frames

    return total_num_frames


def handle_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--video-dir",
        type=Path,
        help="Directory containing videos.",
    )
    args = parser.parse_args()
    return args


def get_frames_in_a_video(video_path: Path | str) -> int:
    """Returns the number of frames in a video."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("Error opening video stream or file")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return num_frames


if __name__ == "__main__":
    main()
