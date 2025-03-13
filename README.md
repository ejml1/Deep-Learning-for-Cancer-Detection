# Introduction

This project forms the artifacts for a BSc Dissertation at the University of St Andrews. The report is provided in the repository under DeepLearningForCancerDetectionReport.pdf

# Dataset & Running Instructions

The dataset has not been included as it is 6.8 GB. It can be downloaded using the IBM Aspera Connect plugin from the following link: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770 

The downloaded dataset should be named “BM_cytomorphology_data” and placed in the submission folder for the following steps. 

The submission contains a Dockerfile to create a container containing all the required dependencies. The following commands can be used to build and run the container. The following code should then be all run on the docker container’s command line:

```bash
docker build -t model .


docker run -v <replace-with-path-to-the-following-directory>/Deep-Learning-for-Cancer-Detection:/Deep-Learning-for-Cancer-Detection -w /Deep-Learning-for-Cancer-Detection --gpus 1 --shm-size=1g -it -p 8888:8888 --rm model

```

In the preprocess directory, run the following command to remove the identified corrupted images from the dataset:

```bash
python DeleteCorrupted.py
```

The dataset can then be split into the train, validation, and test subsets by creating 2 directories for the validation and test subsets and running the following command. For the purpose of training and testing the model for execution, the 2 directories should be named “validation” and “test”: 

```bash
python Split.py
```

This script will then ask for the train (the BM_cytomorphology_data directory), validation, and test directories to be input.

To create reproducible results, data augmentation was not performed on the fly. To augment images, copy or rename the “BM_cytomorphology_data” directory to “BM_cytomorphology_data_augmented” should be created. The following command can then be run (not this will take a long time): 

```bash
python AugmentImages.py
```

To generate explanations to perform the LIME experiment, the following command can be run after creating an explanations/validation directory:

```bash
./CreatePerturbations.sh
```

Before training the optimised model, the following directories should be created. This is used to save the model itself and its training history as a pickle file:

```bash
pickle/augmented
```

The following script can then be run in the docker container’s command line to train the model, note that the expected training directory is “BM_cytomorphology_data_augmented”. Therefore, if the data has not been augmented, it should be renamed to “BM_cytomorphology_data_augmented” anyways.

```bash
python OptimisedModel.py
```

The LIMEResults notebook can be run through JupyterLab in the docker container to produce the LIME experiment results.
