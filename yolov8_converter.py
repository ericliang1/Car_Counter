from ultralytics.data.converter import convert_coco
import os
import shutil

#define paths

train_images_path = "C:/Users/Eric.Jinchen.Liang/fiftyone/coco-2017/train/data"
train_labels_path = "coco_converted/labels/labels"
validation_images_path = "C:/Users/Eric.Jinchen.Liang/fiftyone/coco-2017/validation/data"
validation_labels_path = "coco_converted2/labels/labels"
new_train_images_path = "dataset/images/train" 
new_train_labels_path = "dataset/labels/train"  
new_validation_images_path = "dataset/images/validation"      
new_validation_labels_path = "dataset/labels/validation"

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
    for file in os.listdir(train_images_path):
        shutil.copy(os.path.join(train_images_path, file), new_train_images_path)

    for file in os.listdir(train_labels_path):
        shutil.copy(os.path.join(train_labels_path, file), new_train_labels_path)

    for file in os.listdir(validation_images_path):
        shutil.copy(os.path.join(validation_images_path, file), new_validation_images_path)

    for file in os.listdir(validation_labels_path):
        shutil.copy(os.path.join(validation_labels_path, file), new_validation_labels_path)


if __name__ == "__main__":
    convert_files()
    make_directories()
    move_data()
    convert_label(new_train_labels_path)
    convert_label(new_validation_labels_path)