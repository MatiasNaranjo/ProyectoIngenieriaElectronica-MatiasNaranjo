import re
import time
from pathlib import Path

import libcamera
from picamera2 import Picamera2


def init_camera(resolution=1440):
    """
    Inicializa la cámara PiCamera2 con la configuración deseada.

    Parámetros:
    - max_resolution (bool): Si es True, usa la máxima resolución del sensor.

    Retorna:
    - picam2 (Picamera2): Objeto de cámara ya configurado y listo para capturar.
    """
    picam2 = Picamera2()

    # Configura la cámara para capturar imágenes en formato RGB888 con la resolución deseada
    config = picam2.create_still_configuration(
        main={"format": "RGB888", "size": (resolution, resolution)},
        transform=libcamera.Transform(hflip=1, vflip=1),
    )

    # Aplica la configuración y arranca la cámara
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Espera para estabilizar la imagen

    return picam2


def get_next_session(output_dir: Path, producto: str) -> int:
    output_dir = Path(output_dir)

    pattern = re.compile(rf"{producto}_s(\d+)_\d+\.jpg")
    sessions = []

    for img in output_dir.glob(f"{producto}_s*_*.jpg"):
        match = pattern.match(img.name)
        if match:
            sessions.append(int(match.group(1)))

    return max(sessions, default=0) + 1


def get_next_frame(output_dir: Path, filename_prefix: str, session: int) -> int:
    # Busca el número de frame más alto para el producto y sesión dados, y devuelve el siguiente número disponible

    # Defino formato del nombre de archivo: {filename_prefix}_s{session}_{n_frame}.jpg
    pattern = re.compile(rf"{filename_prefix}_s{session:02d}_(\d+)\.jpg")
    frames = []

    for img in output_dir.glob(f"{filename_prefix}_s{session:02d}_*.jpg"):
        match = pattern.match(img.name)
        if match:
            frames.append(int(match.group(1)))

    return max(frames, default=-1) + 1


def capture_photos(
    picam2, output_dir, filename_prefix, n_photos=40, delay=0.5, session=None
):
    """
    Captura una serie de fotos con la cámara y guarda información de exposición.

    Parámetros:
    - picam2 (Picamera2): Objeto de cámara inicializado.
    - output_dir (str): Carpeta donde se guardarán las fotos.
    - filename_prefix (str): Prefijo para nombrar las fotos.
    - n_photos (int): Cantidad de fotos a capturar.
    - delay (float): Tiempo de espera entre fotos, en segundos.
    - session (int, opcional): Número de sesión manual. Si no se proporciona, se calcula automáticamente.
    """
    # Crea un objeto Path
    output_dir = Path(output_dir)
    # Crea el directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Obtiene el número de sesión siguiente para evitar sobrescribir fotos anteriores
    if session is None:
        session = get_next_session(output_dir, filename_prefix)

    # Obtiene el número de frame siguiente para evitar sobrescribir fotos anteriores
    frame = get_next_frame(output_dir, filename_prefix, session)

    for i in range(n_photos):
        # Construye el nombre del archivo con el formato: {filename_prefix}_s{session}_{n_frame}.jpg
        filename = output_dir / f"{filename_prefix}_s{session:02d}_{frame:04d}.jpg"

        # Captura una imagen
        request = picam2.capture_request()
        request.save("main", filename)  # Guarda la imagen principal

        # Obtiene metadatos útiles de la captura
        metadata = request.get_metadata()
        exposure_time = metadata.get("ExposureTime", "N/A")
        gain = metadata.get("AnalogueGain", "N/A")

        # Muestra por consola información de la captura
        print(f"Foto {i + 1}/{n_photos} -> {filename}")
        print(f"Exposición: {exposure_time} µs | Ganancia: {gain}")

        # Libera el request
        request.release()
        # Incrementa el número de frame para la siguiente foto
        frame += 1
        time.sleep(delay)

    print("Capturas finalizadas.")
    picam2.close()
