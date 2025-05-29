import os

from roboflow import Roboflow

# Ruta absoluta del proyecto
abs_path = "D:/matna/Documents/Escritorio/Facultad/Proyecto"
os.chdir(abs_path)
print(os.getcwd())

# Definir la ruta donde se descargarán los datos del modelo
download_dir = os.path.join(abs_path, "data", "yolo")

# Crear la carpeta si no existe
os.makedirs(download_dir, exist_ok=True)

# Cambiar el directorio actual
os.chdir(download_dir)
print(download_dir)

# Número de versión del dataset en Roboflow
ver = 2
# Autenticación en Roboflow con una clave API
rf = Roboflow(api_key="1BqdoBrABHEibVODyrPg")
# Cargar el proyecto desde Roboflow Workspace
project = rf.workspace("proyecto-final-labels").project("yolo-uifhe")
version = project.version(ver)

dataset = version.download("yolov8")

os.chdir(abs_path)
