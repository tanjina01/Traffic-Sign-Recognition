import os
import cv2
import numpy as np # type: ignore
import pandas as pd
from tqdm import tqdm
import shutil

IMG_SIZE = 48  # Resize images

def create_folder_structure(base_path):
    train_folder = os.path.join(base_path, "GTSRB/train")
    test_folder = os.path.join(base_path, "GTSRB/test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    return train_folder, test_folder


def copy_images_from_csv(csv_path, source_dir, target_base):
    df = pd.read_csv(csv_path)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {csv_path}"):
        class_id = str(row['ClassId'])
        img_path = row['Path'].replace("Train/", "").replace("Test/", "")
        src = os.path.join(source_dir, img_path)
        dst_dir = os.path.join(target_base, class_id)
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(src):
            shutil.copy(src, dst_dir)


def load_images_from_folder(folder):
    X, y = [], []
    for class_id in sorted(os.listdir(folder), key=lambda x: int(x)):
        class_folder = os.path.join(folder, class_id)
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(int(class_id))
    return np.array(X), np.array(y)


def main(base_path="."):
    train_dir = os.path.join(base_path, "Train")
    test_dir = os.path.join(base_path, "Test")
    train_csv = os.path.join(base_path, "Train.csv")
    test_csv = os.path.join(base_path, "Test.csv")

    train_out, test_out = create_folder_structure(base_path)

    print("✅ Organizing images...")
    copy_images_from_csv(train_csv, train_dir, train_out)
    copy_images_from_csv(test_csv, test_dir, test_out)

    print("✅ Loading train images into memory...")
    X_train, y_train = load_images_from_folder(train_out)

    print("✅ Loading test images into memory...")
    X_test, y_test = load_images_from_folder(test_out)

    print("✅ Saving dataset into NumPy format (.npz)...")
    np.savez_compressed(os.path.join(base_path, "gtsrb_dataset.npz"),
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)

    print("✅ Dataset is ready! File saved as gtsrb_dataset.npz")


if __name__ == "__main__":
    main()
