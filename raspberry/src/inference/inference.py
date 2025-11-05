class FrameProcessor:
    def __init__(self, picam2):
        """
        Inicializa el procesador de frames.

        Parámetros:
            picam2: Instancia de la cámara.
        """
        self.picam2 = picam2

    def capture_frame(self):
        """Captura un frame y lo devuelve como un numpy.ndarray."""
        return self.picam2.capture_array()
