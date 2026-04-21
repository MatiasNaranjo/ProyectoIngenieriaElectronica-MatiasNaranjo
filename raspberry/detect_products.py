import logging
import os
import time

from src.camera.camera_utils import init_camera
from src.inference.inference import FrameProcessor
from ultralytics import YOLO

from src.utils.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,  # mostrar mensajes INFO y superiores
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main(config, picam2):
    """
    Script principal de inferencia para Raspberry Pi.

    Este programa captura frames desde la cámara, ejecuta predicciones con un modelo YOLO,
    extrae las detecciones, las registra en logs y guarda imágenes anotadas con bounding boxes.
    Es el punto de entrada de la aplicación de inferencia en la Raspberry Pi.


    Args:
        config: configuración de la aplicación cargada.
        picam2: instancia de la cámara inicializada.
    """
    logging.info("Iniciando aplicación de inferencia")

    os.makedirs(config.inference.inference_dir, exist_ok=True)

    # Cargar el modelo
    model = YOLO(config.inference.model_path)
    logging.info("Modelo cargado, task: %s\n", model.task)

    # Inicializar el procesador de frames
    processor = FrameProcessor(picam2, model)

    # Parámetros de ejecución del bucle de captura
    frame_count = 0  # Contador de frames procesados

    while frame_count < config.inference.max_frames:
        time.sleep(config.inference.frame_delay)  # espera entre frames
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
        processor.draw_detections(
            frame, path=config.inference.inference_dir, frame_count=frame_count
        )


if __name__ == "__main__":
    # Cargar configuración
    config = ConfigLoader(func_name="inference").load()

    # Inicializar la cámara
    picam2 = init_camera(resolution=config.inference.resolution)

    try:
        # Ejecutar el script principal
        main(config, picam2)

    except KeyboardInterrupt:
        logging.info("Detenido por el usuario")

    finally:
        picam2.stop()
