'''
File used to split the data into training, validation, and test sets.
'''

import os
import shutil

def create_class_dirs(data_dir, destination_dir):
    for image_class in os.listdir(data_dir):
        if not os.path.exists(os.path.join(destination_dir, image_class)):
            os.mkdir(os.path.join(destination_dir, image_class))

def create_split(data_dir, destination_dir, split):
    for image_class in os.listdir(data_dir):
        image_class_path = os.path.join(data_dir, image_class)
        _, _, files = next(os.walk(image_class_path))
        number_files = len(files)
        split_number = int(number_files * split)

        for i in range(split_number):
            file_to_move = files[i]
            file_to_move_path = os.path.join(image_class_path, file_to_move)
            destination_path = os.path.join(destination_dir, image_class)
            shutil.move(file_to_move_path, destination_path)

# Move all images into parent class folder
def move_files(data_dir):
    # BM_cytomorphology_data/..
    # Example: BM_cytomorphology_data/ABE
    for image_class in os.listdir(data_dir):
        # BM_cytomorphology_data/../..
        # Example: BM_cytomorphology_data/ABE/ABE_00001.jpg 
        # Example: BM_cytomorphology_data/ART/0001-1000
        image_class_path = os.path.join(data_dir, image_class)
        for maybe_dir in os.listdir(image_class_path):
            maybe_dir_path = os.path.join(image_class_path, maybe_dir)
            if os.path.isdir(maybe_dir_path):
                for image in os.listdir(maybe_dir_path):
                    image_path = os.path.join(maybe_dir_path, image)
                    shutil.move(image_path, image_class_path)

# Remove empty directories from class folders
def remove_empty_dirs(data_dir):
    for image_class in os.listdir(data_dir):
        image_class_path = os.path.join(data_dir, image_class)
        for maybe_dir in os.listdir(image_class_path):
            maybe_dir_path = os.path.join(image_class_path, maybe_dir)
            if os.path.isdir(maybe_dir_path):
                shutil.rmtree(maybe_dir_path)

# Create classes in the test directory
def create_test_dirs(data_dir, test_dir):
    create_class_dirs(data_dir, test_dir)

# Create test split for each class by moving the images from the data directory to the test directory
def create_test_split(data_dir, test_dir):
    create_split(data_dir, test_dir, 0.3)

def create_validation_dirs(data_dir, validation_dir):
    create_class_dirs(data_dir, validation_dir)

def create_validation_split(data_dir, validation_dir):
    create_split(data_dir, validation_dir, 0.2)


def main():
    data_dir = input("Enter the path to the data directory: ")
    test_dir = input("Enter the path to the test directory: ")
    validation_dir = input("Enter the path to the validation directory: ")

    move_files(data_dir)
    remove_empty_dirs(data_dir)
    create_test_dirs(data_dir, test_dir)
    create_test_split(data_dir, test_dir)
    create_validation_dirs(data_dir, validation_dir)
    create_validation_split(data_dir, validation_dir)

if __name__ == "__main__":
    main()