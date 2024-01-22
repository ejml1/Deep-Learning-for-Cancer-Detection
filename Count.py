import os

def main():
    data_dir = input("Enter the path to the data directory: ")
    for image_class in os.listdir(data_dir):
        image_class_path = os.path.join(data_dir, image_class)
        _, _, files = next(os.walk(image_class_path))
        print(image_class, len(files))

if __name__ == "__main__":
    main()