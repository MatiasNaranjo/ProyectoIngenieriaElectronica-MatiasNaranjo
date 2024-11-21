import os

from roboflow import Roboflow

# Set the desired download directory
download_dir = "../../../data/yolo"

# Create the directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Change to the download directory
os.chdir(download_dir)

rf = Roboflow(api_key="1BqdoBrABHEibVODyrPg")
project = rf.workspace("proyecto-final-labels").project("proyecto-final-rvqup")
version = project.version(1)
dataset = version.download("yolov11")
