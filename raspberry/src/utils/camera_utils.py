import os
import time
from datetime import datetime

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

    # Obtiene la resolución del sensor
    sensor_size = picam2.sensor_resolution
    config = picam2.create_still_configuration(
        main={"format": "RGB888", "size": (resolution, resolution)}
    )

    # Aplica la configuración y arranca la cámara
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Espera para estabilizar la imagen

    return picam2


def capture_photos(picam2, output_dir, n_photos=40, delay=0.5):
    """
    Captura una serie de fotos con la cámara y guarda información de exposición.

    Parámetros:
    - picam2 (Picamera2): Objeto de cámara inicializado.
    - output_dir (str): Carpeta donde se guardarán las fotos.
    - n_photos (int): Cantidad de fotos a capturar.
    - delay (float): Tiempo de espera entre fotos, en segundos.
    """
    # Crea el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_photos):
        # Genera un nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"{timestamp}_foto_{i}.jpg")

        # Captura una imagen
        request = picam2.capture_request()
        request.save("main", filename)  # Guarda la imagen principal

        # Obtiene metadatos útiles de la captura
        metadata = request.get_metadata()
        exposure_time = metadata.get("ExposureTime", "N/A")
        gain = metadata.get("AnalogueGain", "N/A")

        # Muestra por consola información de la captura
        print(f"Foto {i+1}/{n_photos} -> {filename}")
        print(f"Exposición: {exposure_time} µs | Ganancia: {gain}")

        # Libera el request
        request.release()
        time.sleep(delay)

    print("Capturas finalizadas.")
    picam2.close()
