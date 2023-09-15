"""Sharpens all images in an input dir and creates an out dir."""

from typing import Optional

# import argparse
import os
import cv2
from preprocess import preprocess_image


inp_dir = "data/images/train"
out_dir = "./out"


def main(argv: Optional[list[str]] = None):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-o',"--outdir",type=str,default = './out',help="output directory")
    # args = parser.parse_args()
    # args.outdir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for filename in os.listdir(inp_dir):
        path = os.path.join(inp_dir, filename)
        img = cv2.imread(path)
        img = preprocess_image(img)
        write_path = os.path.join(out_dir, filename)
        cv2.imwrite(write_path, img)
        # print(write_path)


if __name__ == "__main__":
    main()
