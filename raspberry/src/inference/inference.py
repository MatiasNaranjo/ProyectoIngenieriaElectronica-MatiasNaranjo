class FrameProcessor:
    def __init__(self, picam2, model):
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
        self.imgsz = 1440
        self.conf = 0.5
        self.iou = 0.3
        self.last_results = None

    def capture_frame(self):
        """Captura un frame y lo devuelve como un numpy.ndarray."""
        return self.picam2.capture_array()

    def predict_frame(self, frame):
        """Realiza la predicción usando el modelo YOLO."""
        results = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False
        )

        self.last_results = results[0]  # Guarda la ultima predicción
        return self.last_results
