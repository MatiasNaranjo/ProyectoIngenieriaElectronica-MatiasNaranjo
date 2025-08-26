import os

abs_path = "D:/matna/Documents/Escritorio/Facultad/Proyecto"
os.chdir(abs_path)  # Cambio el directorio

from src.data.data_yolo import descargar_dataset
from src.training.training_yolo import entrenar_yolo

api_key = "1BqdoBrABHEibVODyrPg"
version = 8

descargar_dataset(abs_path, version=version, api_key=api_key)
entrenar_yolo(abs_path=abs_path, version=version)
