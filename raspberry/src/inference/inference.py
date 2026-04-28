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
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        font_thickness = 2  # Grosor para efecto "negrita"
        color_text = (255, 255, 255)  # Blanco para que contraste con el fondo verde
        color_fondo = (0, 255, 0)  # Verde para el fondo del texto (bounding box)

        # Verifica que existan resultados previos de detección
        if self.last_results is not None:
            # Copia el frame original para dibujar las anotaciones
            frame_annoteted = frame.copy()
            boxes = self.last_results
            for box in boxes:
                x1, y1, x2, y2 = map(int, box["bbox"])
                cls_name = box["class_name"]  # Nombre de la clase detectada
                conf = box["confidence"]  # Nivel de confianza
                label = f"{cls_name} {conf:.2f}"

                # Dibuja el bounding box
                cv2.rectangle(frame_annoteted, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if y1 > 35:
                    rect_top, rect_bottom = y1 - 30, y1
                    text_y = y1 - 7

                else:
                    rect_top, rect_bottom = y1, y1 + 30
                    text_y = y1 + 22

                # Para font_scale 0.9, el ancho es ~15px por caracter
                w = len(label) * 16 + 5

                # Dibujar el fondo del texto (rectángulo sólido)
                cv2.rectangle(
                    frame_annoteted,
                    (x1, rect_top),
                    (x1 + w, rect_bottom),
                    color_fondo,
                    -1,
                )

                cv2.putText(
                    frame_annoteted,
                    label,
                    (x1, text_y),
                    font,
                    font_scale,
                    color_text,
                    font_thickness,
                    cv2.LINE_AA,
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
