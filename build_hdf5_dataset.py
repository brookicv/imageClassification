#from kaggle_dogs_vs_cats.config import dogs_vs_cats_config

import sys
sys.path.append("/home/liqiang/git/machineLearningSample")

from config import handpose_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from pyimagesearch.preprocessing import aspectawarepreprocessor
from pyimagesearch.io import hdf5datasetwriter
from imutils import paths
import numpy as np 
import progressbar
import json
import cv2
import os

# grab the paths to the images
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# dataset => training sets,test sets
testSize = int(len(trainPaths) * 0.2)
split = train_test_split(trainPaths,trainLabels,test_size = testSize,
    stratify=trainLabels,random_state=42)
(trainPaths,testPaths,trainLabels,testLabels) = split

# split validation data from training sets, training sets => training sets,validation sets
valSize = int(len(trainPaths) * 0.2)
split = train_test_split(trainPaths,trainLabels,test_size=valSize,
    stratify=trainLabels,random_state=42)
(trainPaths,valPaths,trainLabels,valLabels) = split

# put together
datasets = [("train",trainPaths,trainLabels,config.TRAIN_HDF5),
            ("val",valPaths,valLabels,config.VAL_HDF5),
            ("test",testPaths,testLabels,config.TEST_HDF5)]

aap = aspectawarepreprocessor.AspectAwarePreprocessor(256,256)
(R,G,B) = ([],[],[])

for(dType,paths,labels,outputPath) in datasets:
    print("[INFO] building {}...".format(outputPath))
    writer = hdf5datasetwriter.HDF5DatasetWriter((len(paths),256,256,3),outputPath)

    # initialize the progress bar
    widgets = ["Building Dataset: ",progressbar.Percentage()," ",progressbar.Bar()," ",progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widgets).start()

    for(i , (path,label)) in enumerate(zip(paths,labels)):

        # load the image and process it
        image = cv2.imread(path)
        image = aap.process(image)

        # if we are building the training dataset,then compute the mean of each channel int the image,then update
        # the respective lists
        if dType == "train":
            (b,g,r) = cv2.mean(image)[:3]
            R.append(r)
            B.append(b)
            G.append(g)
        
        # add the image and leable to the HDF5 dataset
        writer.add([image],[label])
        pbar.update(i)
    # close
    pbar.finish()
    writer.close()

print("[INFO] serializing means...")
D = {"R":np.mean(R),"G":np.mean(G),"B":np.mean(B)}
f = open(config.DATASET_MEAN,"w")
f.write(json.dumps(D))
f.close()