import matplotlib
matplotlib.use("Agg")

from config import handpose_config as config 
from pyimagesearch.preprocessing import imagetoarraypreprocessor
from pyimagesearch.preprocessing import simplePreprocessor
from pyimagesearch.preprocessing import patchpreprocessor
from pyimagesearch.preprocessing import meanpreprocessor
from pyimagesearch.callbacks import trainingmonitors
from pyimagesearch.io import hdf5datasetgenerator
from pyimagesearch.nn.conv.deepergooglenet import DeeeperGoogleLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import json
import os


aug = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,
    height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

sp = simplePreprocessor.SimplePreprocessor(64,64)
mp = meanpreprocessor.MeanPreprocessor(means["R"],means["G"],means["B"])
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

batchSize = 32

trainGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.TRAIN_HDF5,batchSize,aug=aug,preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)
valGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.VAL_HDF5,batchSize,aug=aug,preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)

print("[INFO] compiling model...")
model = DeeeperGoogleLeNet.build(64,64,3,config.NUM_CLASSES,reg=0.0002)
opt = Adam(1e-3)
model.compile(loss="categorical_crossentropy",optimize=opt,metrics=["accuracy"])

path = os.path.sep.join([config.OUTPUT_PATH,"{}.png".format(os.getpid())])
callbacks = [trainingmonitors.TrainingMonitor(path)]

model.fit_generator(trainGen.generator(),steps_per_epoch=trainGen.numImages //batchSize,
    validation_data = valGen.generator(),validation_steps=valGen.numImages //batchSize,
    epochs = config.NUM_EPOCHS,
    max_queue_size=batchSize*2,
    callbacks=callbacks,verbose=1)

print("[INFO] serializing model...")
model.save(config.MODEL_PATH,overwrite=True)

trainGen.close()
valGen.close()
