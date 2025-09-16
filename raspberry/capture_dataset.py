from src.utils.camera_utils import capture_photos, init_camera

# Inicializar la cámara
picam2 = init_camera(resolution=1440)

# Directorio donde guardar las fotos
clase = "Te"
output_dir = "output/data/" + clase

# Capturar fotos
capture_photos(picam2, output_dir, n_photos=40, delay=0.5)
