import skimage.segmentation
import os
import numpy as np

loc = 'Explanations/Validation/All'
class_names = ["ABE", "ART", "BAS", "BLA", "EBO", "EOS", "FGC", "HAC", "KSC", "LYI", "LYT", "MMZ", "MON", "MYB", "NGB", "NGS", "NIF", "OTH", "PEB", "PLM", "PMO"]
dict_superpixels = {}
total_superpixels = []
for class_name in class_names:
    print("Checking superpixels for: ", class_name, flush=True)
    dict_superpixels[class_name] = []
    image_class_path = os.path.join(loc, class_name)
    images = os.listdir(image_class_path)
    for i, image_name in enumerate(images):
        image_path = os.path.join(image_class_path, image_name)
        image = skimage.io.imread(image_path)
        superpixels = skimage.segmentation.quickshift(image, kernel_size=4,max_dist=200, ratio=0.2)
        num_superpixels = np.unique(superpixels).shape[0]
        dict_superpixels[class_name].append(num_superpixels)
        total_superpixels.append(num_superpixels)

print("Mean superpixels for all images: ", np.mean(total_superpixels), flush=True)

for class_name in class_names:
    print("Mean superpixels for class ", class_name, ": ", np.mean(dict_superpixels[class_name]), flush=True)
    print("Max superpixels for class ", class_name, ": ", np.max(dict_superpixels[class_name]), flush=True)
    print("Min superpixels for class ", class_name, ": ", np.min(dict_superpixels[class_name]), flush=True)
    print()