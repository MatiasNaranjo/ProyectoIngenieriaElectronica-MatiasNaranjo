import os
from datetime import datetime

import cv2


class FrameProcessor:
    def __init__(self, picam2, model, imgsz=1440, conf=0.5, iou=0.3):
        """
        Inicializa el procesador de frames.

        Parámetros:
            picam2: Instancia de la cámara.
            model: Modelo de Machine Learning para realizar predicciones.
            imgsz (int): Tamaño de la imagen para el modelo.
            conf (float): Umbral de confianza para las predicciones.
            iou (float): Umbral de IoU para las predicciones.
            last_results: Últimos resultados de predicción.
        """
        self.picam2 = picam2
        self.model = model
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.last_results = None

    def capture_frame(self):
        """Captura un frame y lo devuelve como un numpy.ndarray."""
        return self.picam2.capture_array()

    def predict_frame(self, frame):
        """Realiza la predicción usando el modelo y devuelve los resultados procesados."""
        results = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False
        )

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf)
            cls_id = int(box.cls)
            cls_name = self.model.names[cls_id]

            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                }
            )
        self.last_results = detections

        return self.last_results

    def draw_detections(self, frame, path=None, frame_count=0):
        """
        Convierte el frame a formato BGR, dibuja los bounding boxes de las detecciones
        y devuelve el frame anotado. Opcionalmente guarda el frame con anotaciones en disco.

        Args:
            frame (np.ndarray): Imagen en formato RGB.
            path (str, optional): Ruta de la carpeta donde guardar las imágenes anotadas.
            frame_count (int, optional): Contador de frames, usado en el nombre del archivo.

        Returns:
            np.ndarray: Frame anotado en formato BGR.
        """

        # Verifica que existan resultados previos de detección
        if self.last_results is not None:
            # Copia el frame original para dibujar las anotaciones
            frame_annoteted = frame.copy()
            boxes = self.last_results
            for box in boxes:
                x1, y1, x2, y2 = map(int, box["bbox"])
                cls_name = box["class_name"]  # Nombre de la clase detectada
                conf = box["confidence"]  # Nivel de confianza

                # Dibuja el rectángulo
                cv2.rectangle(frame_annoteted, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_annoteted,
                    f"{cls_name} {conf:.2f}",
                    (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 0),
                    2,
                )

            # Si se especificó una ruta, guarda el frame anotado
            if path is not None:
                # Genera un nombre de archivo único con timestamp y número de frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(
                    path, f"{timestamp}_frame_{frame_count:04d}.jpg"
                )

                cv2.imwrite(filename, frame_annoteted)
            return frame_annoteted

        else:
            print("No hay resultados para dibujar.")
