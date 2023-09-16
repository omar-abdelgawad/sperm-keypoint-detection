"""Extract annotations from annotations_file_path and outputs it in out dir in yolo format"""

import os.path
from xml.dom import minidom
import sys

out_dir = "./out"
annotations_file_path = "other_data/annotations/train/annotations.xml"

# Constants
label_dict = {"sperm": 0}
bbox_type = tuple[int, int, int, int]
obj_keypoints_type = list[tuple[int, int]]
bbox_error = 4  # px


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

        try:  # make sure that the number of bboxes and points are the same
            assert len(points) == len(bboxes)
        except AssertionError:
            print(
                f"""mismatch in number of points and boxes in image {name},
                len(points)={len(points)}, len(bboxes)={len(bboxes)}."""
            )
            sys.exit()

        image_keypoints = prepare_keypoints_image(points)

        with open(os.path.join(out_dir, name + ".txt"), "w") as label_file:
            # iterate over pairs of bbox and keypoints for every sperm
            for bbox in bboxes:
                class_index = label_dict[bbox.getAttribute("label")]

                xtl, ytl, xbr, ybr = prepare_bbox(bbox)

                w = xbr - xtl
                h = ybr - ytl
                x_cen_norm = (xtl + (w / 2)) / width
                y_cen_norm = (ytl + (h / 2)) / height
                w_norm = w / width
                h_norm = h / height

                dataset_label = f"{class_index} {x_cen_norm} {y_cen_norm} {w_norm} {h_norm} "  # last space in str is necessary

                cur_obj_keypoints = assign_keypoint_to_bbox(
                    (xtl, ytl, xbr, ybr), image_keypoints
                )
                for coordinate in cur_obj_keypoints:
                    px, py = map(int, coordinate)
                    px_norm, py_norm = px / width, py / height
                    try:  # make sure that the points and boxes are ordered correctly
                        assert (xtl - bbox_error <= px <= xbr + bbox_error) and (
                            ytl - bbox_error <= py <= ybr + bbox_error
                        )
                    except AssertionError:
                        print(
                            f" the point is {px},{py} and {xtl=}, {ytl=}, {xbr=}, {ybr=} in image {name}"
                        )
                    dataset_label += f"{px_norm} {py_norm} "  # last space is necessary, also didn't hardcode visibility as 2
                dataset_label = dataset_label.rstrip() + "\n"
                label_file.write(dataset_label)

    print("Finished writing all label files.")


def prepare_bbox(bbox: minidom.Element) -> bbox_type:
    xtl = int(float(bbox.getAttribute("xtl")))
    ytl = int(float(bbox.getAttribute("ytl")))
    xbr = int(float(bbox.getAttribute("xbr")))
    ybr = int(float(bbox.getAttribute("ybr")))
    return xtl, ytl, xbr, ybr


def prepare_keypoints_single_obj(keypoints: minidom.Element) -> obj_keypoints_type:
    ret: list[tuple[int, int]] = []
    key_points_coordinates = keypoints.getAttribute("points").split(";")
    for coordinate in key_points_coordinates:
        px, py = map(int, map(float, coordinate.split(",")))
        ret.append((px, py))
    return ret


def prepare_keypoints_image(keypoints) -> list[obj_keypoints_type]:
    ret = []
    for keypoints_obj in keypoints:
        ret.append(prepare_keypoints_single_obj(keypoints_obj))
    return ret


def count_num_keypoints_in_bbox(bbox: bbox_type, keypoints: obj_keypoints_type) -> int:
    ret: int = 0
    xtl, ytl, xbr, ybr = bbox
    for px, py in keypoints:
        if (xtl <= px <= xbr) and (ytl <= py <= ybr):
            ret += 1
    return ret


def assign_keypoint_to_bbox(bbox, keypoints) -> obj_keypoints_type:
    sort_func = lambda keypoint_lst: count_num_keypoints_in_bbox(bbox, keypoint_lst)
    keypoints.sort(key=sort_func)
    return keypoints.pop()


if __name__ == "__main__":
    main()
