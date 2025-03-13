#Notebook taken and modified from https://nbviewer.org/url/arteagac.github.io/blog/lime_image.ipynb

import numpy as np
from keras.applications.imagenet_utils import decode_predictions
import skimage.io 
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import time

# Avoid Out of Memory Error by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

folder = 'pickle'
inner_folder = 'augmented'

num_perturb = 128
superpixel_set = [1, 5, 15, 25, 30, 35, 40, 45]


with open(os.path.join(folder, inner_folder, 'model_pickle'), 'rb') as f:
    model = pickle.load(f)
    
class_names = ["ABE", "ART", "BAS", "BLA", "EBO", "EOS", "FGC", "KSC", "LYI", "LYT", "MMZ", "MON", "MYB", "NGB", "NGS", "NIF", "OTH", "PEB", "PLM", "PMO"]

def perturb_image(img, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1 
    perturbed_image = copy.deepcopy(img)
    # Apply the mask to the perturbed image
    perturbed_image = perturbed_image * mask[:, :, np.newaxis]
    # Turn image gray
    perturbed_image += (1 - mask[:, :, np.newaxis]) * 128
    
    # Normalize to the range [0, 1]
    perturbed_image = perturbed_image / 255.0
    return perturbed_image

# Uses only 30% of the images in the dataset
def createImages(image_class):
    image_class_path = os.path.join('validation', image_class)
    images = os.listdir(image_class_path)
    num_files = len(images) * 0.3 + 1
    for i, image_name in enumerate(images):
        if i >= num_files:
            break
        image_path = os.path.join(image_class_path, image_name)
        image = skimage.io.imread(image_path)

        # Quick shift algorithm to segment image into superpixels
        superpixels = skimage.segmentation.quickshift(image, kernel_size=4,max_dist=200, ratio=0.2)
        num_superpixels = np.unique(superpixels).shape[0]
        perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

        predictions = []
            
        perturbed_images = np.array([perturb_image(image, pert, superpixels) for pert in perturbations])
        predictions = model.predict(perturbed_images, verbose=0, batch_size=32)
        gc.collect()

        predictions = np.array(predictions)

        all_enabled = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 
        distances = sklearn.metrics.pairwise_distances(perturbations, all_enabled, metric='cosine').ravel()

        kernel_width = 0.25
        weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function

        class_to_explain = np.argmax(class_names[np.argmax(image_class)])
        simpler_model = LinearRegression()
        
        simpler_model.fit(X=perturbations, y=predictions[:, class_to_explain], sample_weight=weights)
        gc.collect()
        coeff = simpler_model.coef_
        saveImages(image_class, image, image_name, superpixels, num_superpixels, coeff)

def saveImages(image_class, image, image_name, superpixels, num_superpixels, coeff):
    for i in superpixel_set:
        if (i > num_superpixels):
            break
        top_features = np.argsort(coeff)[-i:] 
        mask = np.zeros(num_superpixels) 
        mask[top_features]= True #Activate top superpixels
        perturbed_image = perturb_image(image, mask, superpixels)
        path = os.path.join('explanations', 'validation', str(i), image_class)
        if not os.path.exists(path):
            os.makedirs('explanations/validation/' + str(i) + '/' + image_class)
        plt.imsave(os.path.join(path, image_name), perturbed_image)
    path = os.path.join('explanations', 'validation', 'all', image_class)
    if not os.path.exists(path):
        os.makedirs('explanations/validation/all/' + image_class)
    plt.imsave(os.path.join(path, image_name), image)


def main():
    data_dir = input("Enter Class Name To Perturb Images: ")
    start_time = time.time()
    createImages(data_dir)
    end_time = time.time()
    print("Done!")
    print("Time taken in seconds: ", end_time - start_time)

if __name__ == "__main__":
    main()