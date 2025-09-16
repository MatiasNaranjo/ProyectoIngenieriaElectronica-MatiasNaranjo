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

folders_down = ["output"]

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
