import os
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib.request
from PIL import Image

os.makedirs("dataset/images/train", exist_ok=True)
os.makedirs("dataset/images/validation", exist_ok=True)
os.makedirs("dataset/labels/train", exist_ok=True)
os.makedirs("dataset/labels/validation", exist_ok=True)

urls = pd.read_csv("data.csv")["Image_url"]

train, val = train_test_split(urls, test_size=0.1, random_state=42)

for folder, urls in zip(["train", "validation"], [train, val]):
    for i, url in enumerate(urls, 1):
        try:
            original = urllib.request.urlopen(url)
            img = Image.open(original)
            img = img.convert("RGB")
            img.save(f"dataset/images/{folder}/image_{i}.jpg", "JPEG")

            with open(f"dataset/labels/{folder}/image_{i}.txt", "w") as label_file:
                label_file.write("0 0.5 0.5 1.0 1.0\n")
        except:
            pass
