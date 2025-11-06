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
