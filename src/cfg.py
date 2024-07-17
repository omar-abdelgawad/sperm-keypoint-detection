"""Includes all the constants, and configurations used in the project."""
from itertools import cycle

import cv2

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
POINTS_1_TO_4_COLOR = GREEN
POINTS_5_TO__COLOR = BLUE
PROJECTION_LINE_COLOR = BLUE
ALLOWED_VIDEO_EXTENSIONS = (
    ".asf",
    ".avi",
    ".gif",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".wmv",
    ".webm",
)
MAGNIFICATION_LIST = ("5X", "10X", "40X", "63X")
PIXEL_SIZE_MICRO = (1.7, 0.85, 0.22, 0.136)
PIXEL_SIZE_FOR_MAGNIFICATION = {
    k: v for k, v in zip(MAGNIFICATION_LIST, PIXEL_SIZE_MICRO)
}
