import sys
from enum import Enum
from pathlib import Path
import random
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data.preprocessing.utils import resize_image

random.seed(42)

__all__ = ["CholecSeg8k"]


class CholecSeg8k:
    class ClassId(Enum):
        BLACK_BACKGROUND = 0
        ABDOMINAL_WALL = 1
        LIVER = 2
        GASTROINTESTINAL_TRACT = 3
        FAT = 4
        GRASPER = 5
        CONNECTIVE_TISSUE = 6
        BLOOD = 7
        CYSTIC_DUCT = 8
        L_HOOK_ELECTROCAUTERY = 9
        GALLBLADDER = 10
        HEPATIC_VEIN = 11
        LIVER_LIGAMENT = 12

    class Color(Enum):
        BLACK_BACKGROUND = (127, 127, 127)
        ABDOMINAL_WALL = (140, 140, 210)
        LIVER = (114, 114, 255)
        GASTROINTESTINAL_TRACT = (156, 70, 231)
        FAT = (75, 183, 186)
        GRASPER = (0, 255, 170)
        CONNECTIVE_TISSUE = (0, 85, 255)
        BLOOD = (0, 0, 255)
        CYSTIC_DUCT = (0, 255, 255)
        L_HOOK_ELECTROCAUTERY = (184, 255, 169)
        GALLBLADDER = (165, 160, 255)
        HEPATIC_VEIN = (128, 50, 0)
        LIVER_LIGAMENT = (0, 74, 111)

    def __init__(self, root: str, anno_mode: str = "all") -> None:
        self.root = Path(root)
        self.path_raw = self.root / "raw"
        self.path_processed = self.root / "processed"
        self.path_processed.mkdir(parents=True, exist_ok=True)
        self.anno_mode = anno_mode
        #
        self.list_path_filenames = []

    def load_data(self) -> None:
        # get list of file names of images
        for subdir in sorted(self.path_raw.iterdir()):
            for imgdir in sorted(subdir.iterdir()):
                for path_img in sorted(imgdir.iterdir()):
                    if "mask" not in path_img.name:
                        self.list_path_filenames.append(str(path_img))

    def preprocess(self) -> None:
        print("Preprocessing CholecSeg8k dataset...")
        # load labels
        list_labels = []
        list_images = []

        print("# Relabeling data...")
        if self.anno_mode == "tissue":
            for i, path in tqdm(enumerate(self.list_path_filenames), total=len(self.list_path_filenames), desc="Label", file=sys.stdout):
                label = cv2.imread(path[:-4] + "_color_mask.png", cv2.IMREAD_COLOR)
                mask = np.zeros_like(label, dtype=label.dtype)
                mask[np.all(label == self.Color.ABDOMINAL_WALL.value, axis=-1)] = (self.ClassId.ABDOMINAL_WALL.value, ) * 3
                mask[np.all(label == self.Color.LIVER.value, axis=-1)] = (self.ClassId.LIVER.value, ) * 3
                mask[np.all(label == self.Color.GASTROINTESTINAL_TRACT.value, axis=-1)] = (self.ClassId.GASTROINTESTINAL_TRACT.value, ) * 3
                mask[np.all(label == self.Color.FAT.value, axis=-1)] = (self.ClassId.FAT.value, ) * 3
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = resize_image(mask, img_size=256)
                list_labels.append(mask)
        list_labels = np.array(list_labels)

        print("# Loading images...")
        for path in tqdm(self.list_path_filenames, total=len(self.list_path_filenames), desc="Image", file=sys.stdout):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_image(image, img_size=256)
            list_images.append(image)
        list_images = np.array(list_images, dtype=np.float32) / 255.

        # split train/val/test sets with the ratio of 20% : 64% : 16%
        print("# splitting train, val, test sets...")
        list_images_trainval, list_images_test, list_labels_trainval, list_labels_test, list_path_images_trainval, list_path_images_test = train_test_split(
            list_images, list_labels, self.list_path_filenames,
            test_size=0.2,
            random_state=42
        )

        list_images_train, list_images_val, list_labels_train, list_labels_val, list_path_images_train, list_path_images_val = train_test_split(
            list_images_trainval, list_labels_trainval, list_path_images_trainval,
            test_size=0.2,
            random_state=42
        )

        print(f"Total of train images/labels: {len(list_images_train)}/{len(list_labels_train)}")
        print(f"Total of val images/labels: {len(list_images_val)}/{len(list_labels_val)}")
        print(f"Total of test images/labels: {len(list_images_test)}/{len(list_labels_test)}")

        print("# Computing Mean and Std on trainval sets...")
        sum_pixels = np.zeros(3, dtype=np.float32)
        sum_squared_pixels = np.zeros(3, dtype=np.float32)
        num_pixels = float(list_images_trainval.shape[0] * list_images_trainval.shape[1] * list_images_trainval.shape[2])
        for image in tqdm(list_images_trainval, total=list_images_trainval.shape[0], file=sys.stdout):
            sum_pixels += np.sum(image, axis=(0, 1))
            sum_squared_pixels += np.sum(image ** 2, axis=(0, 1))
        mean = sum_pixels / num_pixels
        std = np.sqrt((sum_squared_pixels / num_pixels) - (mean ** 2))

        print(f"Mean value: {mean}")
        print(f"Std value: {std}")
        np.savez(str(Path(self.path_processed) / "CholecSeg8k.npz"), mean=mean, std=std)

        print("# Normalizing data...")
        list_images_train = (list_images_train - mean) / std
        list_images_val = (list_images_val - mean) / std
        list_images_test = (list_images_test - mean) / std

        print("Saving to data...")
        np.savez(str(self.path_processed / "data_train.npz"), image=list_images_train, label=list_labels_train, name=list_path_images_train)
        np.savez(str(self.path_processed / "data_val.npz"), image=list_images_val, label=list_labels_val, name=list_path_images_val)
        np.savez(str(self.path_processed / "data_test.npz"), image=list_images_test, label=list_labels_test, name=list_path_images_test)

        del list_images_train, list_images_val, list_images_test, list_images_trainval
        del list_labels_train, list_labels_val, list_labels_test, list_labels_trainval
        del list_path_images_train, list_path_images_val, list_path_images_test, list_path_images_trainval
        del list_images, list_labels
        print("Preprocessing done!")
