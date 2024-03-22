import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from skimage import io
from PIL import Image

train_datagen = ImageDataGenerator(
    rotation_range=180,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    fill_mode='reflect',
)


# Adapted from https://github.com/bnsreenu/python_for_microscopists/blob/master/127_data_augmentation_using_keras.py

for image_class in os.listdir('BM_cytomorphology_data_augmented'):
    dataset = []
    image_class_path = os.path.join('BM_cytomorphology_data_augmented', image_class)
    
    _, _, files = next(os.walk(image_class_path))
    number_files = len(files)

    images_needed = 25000 - number_files
        
    if images_needed > 0:
        print('Augmenting images for: ' + image_class)
        images = os.listdir(image_class_path)
        for i, image_name in enumerate(images):
            if not image_name.startswith('aug'):
                image_path = os.path.join(image_class_path, image_name)
                image = io.imread(image_path)
                image = Image.fromarray(image, 'RGB')
                dataset.append(np.array(image))
        
        x = np.array(dataset)

        for batch in train_datagen.flow(x, batch_size=64,  
                                        save_to_dir=image_class_path, 
                                        save_prefix='aug', 
                                        save_format='jpg'):
            
            _, _, files = next(os.walk(image_class_path))
            number_files = len(files)
            if (number_files >= 25000):
                print('Finished augmenting images for: ' + image_class)
                break

for image_class in os.listdir('BM_cytomorphology_data_augmented'):
    image_class_path = os.path.join('BM_cytomorphology_data_augmented', image_class)
    
    _, _, files = next(os.walk(image_class_path))
    number_files = len(files)
    
    print(f"{image_class}: {number_files} images")