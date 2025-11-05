class FrameProcessor:
    def __init__(self, picam2, model):
        """
        Inicializa el procesador de frames.

        Parámetros:
            picam2: Instancia de la cámara.
            model: Modelo de Machine Learning para realizar predicciones.
        """
        self.picam2 = picam2
        self.model = model

    def capture_frame(self):
        """Captura un frame y lo devuelve como un numpy.ndarray."""
        return self.picam2.capture_array()

    def predict_frame(self, frame):
        """Realiza la predicción usando el modelo YOLO."""
        return self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False
        )
