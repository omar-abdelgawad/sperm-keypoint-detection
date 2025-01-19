"""Run script one time to extract dataset of images from VIDEO_PATH and get its annotations in a .txt file."""

import cv2
import os
import numpy as np
from random import shuffle
from ultralytics import YOLO
from typing import TextIO
from itertools import cycle
from pathlib import Path

# from preprocess import preprocess_image

# TODO: move video from good_quality to used_in_training
VIDEO_PATH = Path("other_data/sperm_vids/good_quality")
TRAIN_PATH = Path("data/images/train_2")
VAL_PATH = Path("data/images/val_2")
ANNOTATIONS_PATH = Path("other_data/annotations")
RATE = 20
VAL_PERCENTAGE = 0.15
NUM_VAL_IMAGES = 75
NUM_TRAIN_IMAGES = 425
DATASET_OFFSET = len(os.listdir(TRAIN_PATH)) + len(os.listdir(VAL_PATH))
MODEL_PATH = Path("runs/pose/train5/weights/last.pt")


def open_image_tag(file: TextIO, id:int, name:str, width:float, height:float) -> None:
    file.write(f'<image id="{id}" name="{name}" width="{width}" height="{height}">\n')


def close_image_tag(file: TextIO):
    file.write("</image>\n")


def write_box_tag(file: TextIO, results) -> bool:
    if not np.all(results.boxes.xyxy.shape):
        return False
    for two_pts_tensor in results.boxes.xyxy:
        xtl, ytl, xbr, ybr = two_pts_tensor
        file.write(
            f'\t<box label="sperm" source="file" occluded="0" xtl="{xtl}" ytl="{ytl}" xbr="{xbr}" ybr="{ybr}" z_order="0">\n'
        )
        file.write("\t</box>\n")
    return True


def write_points_tag(file: TextIO, results) -> bool:
    if not np.all(results.keypoints.xy.shape):
        return False
    for obj in results.keypoints.xy:
        points = ""
        for p1, p2 in obj:
            points += f"{p1:.2f},{p2:.2f};"
        points = points[:-1]  # remove last semicolon
        file.write(
            f'\t<points label="sperm" source="file" occluded="0" points="{points}" z_order="0">\n'
        )
        file.write("\t</points>\n")
    return True


def main():
    model = YOLO(MODEL_PATH)
    filenames_set = set()
    frame_count = 0
    img_count = 0
    filenames = os.listdir(VIDEO_PATH)
    shuffle(filenames)
    with open(ANNOTATIONS_PATH / "val.txt", "w") as val_file, \
    open(ANNOTATIONS_PATH / "train.txt", "w") as train_file:
        for filename in cycle(filenames):
            if img_count >= NUM_TRAIN_IMAGES + NUM_VAL_IMAGES:
                break
            print(f"opened {filename}")
            path = VIDEO_PATH / filename
            video = cv2.VideoCapture(path)
            while img_count < NUM_TRAIN_IMAGES + NUM_VAL_IMAGES:
                flag, frame = video.read()
                if not flag:
                    break
                frame_count += 1
                if frame_count % RATE != 0:
                    continue
                img_count += 1
                print(f"\t{img_count}")
                filenames_set.add(filename)
                # open two tags
                parent_path = VAL_PATH if img_count <= NUM_VAL_IMAGES else TRAIN_PATH
                file_to_write = val_file if img_count <= NUM_VAL_IMAGES else train_file

                image_name = (
                    os.path.splitext(filename)[0]
                    + f"_{img_count + DATASET_OFFSET}"
                    + ".jpeg"
                )
                results = model(source=frame, show=False, save=False)[0]

                # image tag variables
                id = img_count + DATASET_OFFSET
                name = image_name
                width = frame.shape[1]
                height = frame.shape[0]
                open_image_tag(
                    file_to_write, id=id, name=name, width=width, height=height
                )
                write_box_tag(file_to_write, results)
                write_points_tag(file_to_write, results)
                close_image_tag(file_to_write)
                write_path = os.path.join(parent_path, image_name)
                cv2.imwrite(write_path, frame)

    print(f"Finished Succesfully collecting data from {len(filenames_set)} videos")


if __name__ == "__main__":
    main()
