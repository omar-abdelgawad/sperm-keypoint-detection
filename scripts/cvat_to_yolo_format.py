"""Extract annotations from annotations_file_path and outputs it in out dir in yolo format"""

import os.path
from xml.dom import minidom

out_dir = "./out"
annotations_file_path = "other_data/annotations/val/annotations.xml"

# Constants
label_dict = {"sperm": 0}


def main():
    # create out dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file = minidom.parse(annotations_file_path)
    images = file.getElementsByTagName("image")

    for image in images:
        width = int(image.getAttribute("width"))
        height = int(image.getAttribute("height"))
        name = os.path.splitext(image.getAttribute("name"))[0]  # first ele is file name
        points = image.getElementsByTagName("points")
        bboxes = image.getElementsByTagName("box")
        with open(os.path.join(out_dir, name + ".txt"), "w") as label_file:
            # iterate over pairs of bbox and keypoints for every sperm
            for keypoints, bbox in zip(points, bboxes):
                class_index = label_dict[keypoints.getAttribute("label")]

                xtl = float(bbox.getAttribute("xtl"))
                ytl = float(bbox.getAttribute("ytl"))
                xbr = float(bbox.getAttribute("xbr"))
                ybr = float(bbox.getAttribute("ybr"))
                w = xbr - xtl
                h = ybr - ytl

                x_cen_norm = (xtl + (w / 2)) / width
                y_cen_norm = (ytl + (h / 2)) / height
                w_norm = w / width
                h_norm = h / height

                dataset_label = f"{class_index} {x_cen_norm} {y_cen_norm} {w_norm} {h_norm} "  # last space in str is necessary

                keypoints = keypoints.getAttribute("points").split(";")
                for coordinate in keypoints:
                    px, py = map(float, coordinate.split(","))
                    px_norm, py_norm = px / width, py / height
                    try:  # make sure that the points and boxes are ordered correctly
                        assert (xtl <= px <= xbr) and (ytl <= py <= ybr)
                    except AssertionError:
                        print(
                            f" the point is {px},{py} and {xtl=}, {ytl=}, {xbr=}, {ybr=} in image {name}"
                        )
                    dataset_label += f"{px_norm} {py_norm} "  # last space is necessary, also didn't hardcode visibility as 2
                dataset_label = dataset_label.rstrip() + "\n"
                label_file.write(dataset_label)


if __name__ == "__main__":
    main()
