from ultralytics.data.converter import convert_coco
import os
import shutil

#define paths
TRAIN_IMAGES_PATH = "C:/Users/Eric.Jinchen.Liang/fiftyone/coco-2017/train/data"
TRAIN_LABELS_PATH = "coco_converted/labels/labels"
VALIDATION_IMAGES_PATH = "C:/Users/Eric.Jinchen.Liang/fiftyone/coco-2017/validation/data"
VALIDATION_LABELS_PATH = "coco_converted2/labels/labels"
NEW_TRAIN_IMAGES_PATH = "dataset/images/train"
NEW_TRAIN_LABELS_PATH = "dataset/labels/train"
NEW_VALIDATION_IMAGES_PATH = "dataset/images/validation"
NEW_VALIDATION_LABELS_PATH = "dataset/labels/validation"


#convert labels from coco format to yolo format
def convert_files():
    convert_coco(  
        labels_dir="../../../fiftyone/coco-2017/train",
        use_segments=False,
        use_keypoints=False,
        cls91to80=True,
    )

    convert_coco(  
        labels_dir="../../../fiftyone/coco-2017/validation",
        use_segments=False,
        use_keypoints=False,
        cls91to80=True,
    )


#create dataset directory
def make_directories():
    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/images/validation", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)
    os.makedirs("dataset/labels/validation", exist_ok=True)


#convert labels to usable format
def convert_label(path):
    for label_file in os.listdir(path):
        if label_file.endswith(".txt"): 
            file_path = os.path.join(path, label_file)
            new = []
            with open(file_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) > 0 and parts[0] == "62": 
                        parts[0] = "0" 
                        new.append(" ".join(parts))

            with open(file_path, "w") as file:
                file.write("\n".join(new))


#move data to new dataset directory
def move_data():
    for file in os.listdir(TRAIN_IMAGES_PATH):
        shutil.copy(os.path.join(TRAIN_IMAGES_PATH, file), NEW_TRAIN_IMAGES_PATH)

    for file in os.listdir(TRAIN_LABELS_PATH):
        shutil.copy(os.path.join(TRAIN_LABELS_PATH, file), NEW_TRAIN_LABELS_PATH)

    for file in os.listdir(VALIDATION_IMAGES_PATH):
        shutil.copy(os.path.join(VALIDATION_IMAGES_PATH, file), NEW_VALIDATION_IMAGES_PATH)

    for file in os.listdir(VALIDATION_LABELS_PATH):
        shutil.copy(os.path.join(VALIDATION_LABELS_PATH, file), NEW_VALIDATION_LABELS_PATH)

if __name__ == "__main__":
    convert_files()
    make_directories()
    move_data()
    convert_label(NEW_TRAIN_LABELS_PATH)
    convert_label(NEW_VALIDATION_LABELS_PATH)