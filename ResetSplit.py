import os
import shutil

def reset_files(source_dir, original_dir):
    for image_class in os.listdir(source_dir):
        image_class_path = os.path.join(source_dir, image_class)
        for image in os.listdir(image_class_path):
            image_path = os.path.join(image_class_path, image)
            original_class_dir = os.path.join(original_dir, image_class)
            shutil.move(image_path, original_class_dir)

def main():
    source_dir = input("Enter the path to the directory you want to empty: ")
    original_dir = input("Enter the path to the original directory: ")

    reset_files(source_dir, original_dir)

if __name__ == "__main__":
    main()