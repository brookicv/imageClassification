
#define the paths to the image directory
IMAGES_PATH = "datasets/handpose"

NUM_CLASSES = 7
NUM_VAL_IMAGES = 600
NUM_TEST_IMAGES = 600

NUM_EPOCHS = 10

# define the path to the output training,validation,and testing HDF5 files
TRAIN_HDF5 = "datasets/hdf5/train.hdf5"
VAL_HDF5 = "datasets/hdf5/val.hdf5"
TEST_HDF5 = "datasets/hdf5/test.hdf5"

# path to output model file
MODEL_PATH = "output/alexnet_handpose.model"

# define the path to the dataset mean 
DATASET_MEAN = "output/handpose.json"

# define the path to the output directory used for storing plots,classification,etc.
OUTPUT_PATH = "output"