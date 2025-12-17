import os
import sys

# Agrega el directorio raíz al path
BASE_DIR = r"D:\matna\Documents\Escritorio\Facultad\Proyecto"
sys.path.append(BASE_DIR)

from src.utils.files import exportar_a_raspberry, importar_de_raspberry

# Variables
LOCAL_DIR = r"D:\matna\Documents\Escritorio\Facultad\Proyecto\Raspberry"
RASPI_DIR = "/home/matna/proyecto"
RASPI_USER = "matna"
RASPI_IP = "raspberrypi"
key_path = "C:/Users/matna/.ssh/id_ed25519"
passphrase = "naranjo6201919"

files_up = ["main_raspi.py", "capture_dataset.py"]
folders_up = ["model", "src"]
folders_down = ["output"]

# Copiar archivos de la PC a Raspberry PI
exportar_a_raspberry(
    LOCAL_DIR,
    RASPI_DIR,
    RASPI_USER,
    RASPI_IP,
    files_up,
    folders_up,
    key_path,
    passphrase,
)

# Copiar archivos de RaspberryPI a la PC
importar_de_raspberry(
    RASPI_DIR,
    LOCAL_DIR,
    RASPI_USER,
    RASPI_IP,
    folders_down,
    key_path,
    passphrase,
)
