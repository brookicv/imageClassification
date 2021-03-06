from config import handpose_config as config
from pyimagesearch.preprocessing import simplePreprocessor
from pyimagesearch.preprocessing import meanpreprocessor
from pyimagesearch.preprocessing import croppreprocessor
from pyimagesearch.preprocessing import imagetoarraypreprocessor
from pyimagesearch.io import hdf5datasetgenerator
from pyimagesearch.utils.ranked import rank5_accuracy
from keras.models import load_model
import numpy as np 
import progressbar
import json 

means = json.loads(open(config.DATASET_MEAN).read())

sp = simplePreprocessor.SimplePreprocessor(227,227)
mp = meanpreprocessor.MeanPreprocessor(means["R"],means["G"],means["B"])
cp = croppreprocessor.CropPreprocessor(227,227)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

print("[INFO] loading model ...")
model = load_model(config.MODEL_PATH)

print("[INFO] predicting on test data(no crops)...")
testGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.TEST_HDF5,64,preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)

predictions = model.predict_generator(testGen.generator(),steps=testGen.numImages // 64,max_queue_size=64*2)

(rank1,_)=rank5_accuracy(predictions,testGen.db["labels"])
print("[INFO] rank-1:{:.2f}%".format(rank1 * 100))
testGen.close()

testGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.TEST_HDF5,64,preprocessors=[mp],classes=config.NUM_CLASSES)
predictions = []

widgets = ["Evaluating: ",progressbar.Percentage()," ",progressbar.Bar()," ",progressbar.ETA()]
pbar = progressbar.ProgressBar(max_value=testGen.numImages // 64,widgets=widgets).start()

for(i,(images,labels)) in enumerate(testGen.generator(passes=1)):

    for image in images:
        
        crops = cp.process(image)
        crops = np.array([iap.process(c) for c in crops],dtype="float32")

        pred = model.predict(crops)
        predictions.append(pred.mean(axis=0))
    pbar.update(i)
pbar.finish()

print("[INFO] predicting on test data(with crops)...")
(rank1,_) = rank5_accuracy(predictions,testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()
