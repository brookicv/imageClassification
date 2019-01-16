
import cv2
import os 
import argparse
from pyimagesearch.preprocessing import aspectawarepreprocessor

ap = argparse.ArgumentParser()

ap.add_argument("-sf","--source",required=True,help="Source folder to process")
ap.add_argument("-df","--destination",required=True,help="Destination to save")
args = vars(ap.parse_args())


sourceFolder = args["source"]
destination = args["destination"]

if os.path.exists(destination) != True:
    os.mkdir(destination)

aap = aspectawarepreprocessor.AspectAwarePreprocessor(256,256)

if os.path.exists(sourceFolder) == True:
    os.chdir(sourceFolder)
    folders = os.listdir(sourceFolder)
    
    

    for folder in folders:
        if folder[0] == ".":
            continue

        df = os.path.join(destination,folder)
        if os.path.exists(df) != True:
            os.mkdir(df)

        for (file,i) in zip(os.listdir(folder),range(0,len(os.listdir(folder)))):
            if file[0] == ".":
                continue
            if os.path.splitext(file)[-1] != ".jpg":
                continue
            imagePath = os.path.join(sourceFolder,folder,file)
            print(imagePath)
            im = cv2.imread(imagePath)
            im = aap.process(im)
           
            cv2.imwrite(os.path.join(destination,folder,str(i) + ".jpg"),im)
            