import logging
import os
import time

from src.inference.inference import FrameProcessor
from src.utils.camera_utils import init_camera
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,  # mostrar mensajes INFO y superiores
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Carpeta donde se guardarán las imágenes
SAVE_DIR = "/home/matna/proyecto/output/inference_result"

# Modelo que realiza la inferencia
MODEL_PATH = "/home/matna/proyecto/model/raspi_model.pt"


def main(picam2):
    """
    Script principal de inferencia para Raspberry Pi.

    Este programa captura frames desde la cámara, ejecuta predicciones con un modelo YOLO,
    extrae las detecciones, las registra en logs y guarda imágenes anotadas con bounding boxes.
    Es el punto de entrada de la aplicación de inferencia en la Raspberry Pi.


    Args:
        picam2: instancia de la cámara inicializada.
    """
    logging.info("Iniciando aplicación de inferencia")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Cargar el modelo
    model = YOLO(MODEL_PATH)
    logging.info("Modelo cargado, task: %s\n", model.task)

    # Inicializar el procesador de frames
    processor = FrameProcessor(picam2, model)

    # Parámetros de ejecución del bucle de captura
    frame_count = 0  # Contador de frames procesados
    max_frames = 2  # Cantidad de frames a procesar
    frame_delay = 1  # Tiempo entre capturas (segundos)

    while frame_count < max_frames:
        time.sleep(frame_delay)  # espera entre frames
        frame_count += 1
        logging.info(f"Procesando frame {frame_count}...")

        # Capturar frame en RGB
        frame = processor.capture_frame()

        # Realizar predicción
        results = processor.predict_frame(frame)

        # Registrar resultados
        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            logging.info(
                "Clase: %d - %s | Bounding Box: (%.1f, %.1f, %.1f, %.1f)",
                result["class_id"],
                result["class_name"],
                x1,
                y1,
                x2,
                y2,
            )

        logging.info("Total de productos detectados: %d\n", len(results))

        # Dibujar y guardar detecciones
        processor.draw_detections(frame, path=SAVE_DIR, frame_count=frame_count)


if __name__ == "__main__":
    # Inicializar la cámara
    picam2 = init_camera(resolution=1440)

    try:
        # Ejecutar el script principal
        main(picam2)

    except KeyboardInterrupt:
        logging.info("Detenido por el usuario")

    finally:
        picam2.stop()
