from src.camera.camera_utils import capture_photos, init_camera

from src.utils.config_loader import ConfigLoader


def main():
    # Cargar la configuración
    config = ConfigLoader(func_name="capture").load()

    # Inicializar la cámara
    picam2 = init_camera(resolution=config.cam.resolution)

    # Directorio donde guardar las fotos
    output_dir = config.cam.output_dir + "/" + config.cam.clase

    # Capturar fotos
    capture_photos(
        picam2, output_dir, n_photos=config.cam.n_photos, delay=config.cam.delay
    )


if __name__ == "__main__":
    main()
